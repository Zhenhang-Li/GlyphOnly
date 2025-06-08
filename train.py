
import os
import cv2
import math
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from packaging import version
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import HfFolder, Repository, create_repo, whoami
import datasets
from dataset import DiffData

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import CLIPVisionModel, AutoProcessor, CLIPTextModel, CLIPTokenizer

import transformers
from abinet import MultiLosses,prepare_label,create_ocr_model,preprocess
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from termcolor import colored

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--character_aware_loss_lambda",
        type=float,
        default=0.01,
        help="Lambda for the character-aware loss",
    )
    parser.add_argument(
        "--reg_loss_lambda",
        type=float,
        default=0.01,
        help="Lambda for the character-aware loss",
    )
    parser.add_argument(
        "--character_aware_loss_ckpt",
        type=str,
        default='./checkpoint/character_aware_loss_unet.pth',
        help="The checkpoint for unet providing the charactere-aware loss."
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--no_pos_con",
        action="store_true",
        default=False,
        help="If it is activated, the position and the content of character are not avaible during training.",
    )
    parser.add_argument(
        "--no_con",
        action="store_true",
        default=False,
        help="If it is activated, the content of character is not avaible during training.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--segmentation_mask_aug",
        action="store_true",
        help="Whether to augment the segmentation masks (inspired by https://arxiv.org/abs/2211.13227)."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mask_all_ratio",
        type=float,
        default=0.9,
        help="The training ratio of two branches."
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0,
        help="The scale of noise offset."
    )
    parser.add_argument(
        "--vis_num",
        type=int,
        default=8,
        help="The number of images to be visualized during training."
    )
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=500,
        help="The interval for visualization."
    )

    args = parser.parse_args()

    print('***************')
    print(args)
    print('***************')

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    args.seed = random.randint(0, 1000000) if args.seed is None else args.seed

    print(f'{colored("[√]", "green")} Arguments are loaded.')
    print(args)

    set_seed(args.seed)
    print(f'{colored("[√]", "green")} Seed is set to {args.seed}.')

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            print(args.output_dir)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",cache_dir = "/data1/lzh/tmp/glyphonly/models--runwayml--stable-diffusion-v1-5/"
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",cache_dir = "/data1/lzh/tmp/glyphonly/models--runwayml--stable-diffusion-v1-5/"
    )
    unet = UNet2DConditionModel.from_pretrained(
        "./checkpoint/diffusion_backbone/", subfolder="unet"
    )
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14",cache_dir ='/data1/lzh/tmp/glyphonly/models--openai--clip-vit-large-patch14/')
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14",cache_dir='/data1/lzh/tmp/glyphonly/models--openai--clip-vit-large-patch14/')
    charset,reg = create_ocr_model()
    # Freeze vae
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    reg.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    for name, p in unet.named_parameters():
        if ("transformer_blocks" in name) and ("attn2" in name) and (("to_k" in name) or ("to_v" in name)):
            p.requires_grad_(True)
        elif "conv_in" in name:
            p.requires_grad_(True)
        elif "word_embedding" in name:
            p.requires_grad_(True)
        elif "segmap_conv" in name:
            p.requires_grad_(True)
        elif "first_conv" in name:
            p.requires_grad_(True)
        elif "proj_img" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    train_dataset = DiffData(root_dir="/data1/lzh/glyph_train_data_eee/",mask_all_ratio=args.mask_all_ratio)

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, args.max_train_steps)

    def collate_fn(batch):
        images = torch.stack([example[0] for example in batch])
        images = images.to(memory_format=torch.contiguous_format).float()
        image_masks = torch.cat([example[1].unsqueeze(0) for example in batch], 0)
        segmentation_masks = torch.cat(
            [torch.from_numpy(example[2]).unsqueeze(0).unsqueeze(0) for example in batch], dim=0)
        glyphs =[example[3] for example in batch]
        glyph_tensors = processor(images=glyphs, return_tensors="pt")
        bgs = [example[4] for example in batch]
        bg_tensors = processor(images=bgs, return_tensors="pt")
        text = [example[5] for example in batch]
        point = [example[6] for example in batch]
        bg_in = torch.stack([example[7] for example in batch])
        bg_in = bg_in.to(memory_format=torch.contiguous_format).float()
        return {"images": images,"image_masks": image_masks, "segmentation_masks": segmentation_masks, "glyphs": glyph_tensors,'bgs':bg_tensors,'texts':text,'points':point,'bg_in':bg_in}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )


    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device)
    reg.to(accelerator.device)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    ce_criterion = torch.nn.CrossEntropyLoss()

    from text_segmenter.unet import UNet
    segmenter = UNet(4, 96, True).cuda()
    state_dict = torch.load(args.character_aware_loss_ckpt, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    segmenter.load_state_dict(new_state_dict)
    segmenter.eval()

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet):
                # Convert images to latent space
                features = vae.encode(batch["images"].to(weight_dtype)).latent_dist.sample()
                features = features * vae.config.scaling_factor
                bg_features = vae.encode(batch["bg_in"].to(weight_dtype)).latent_dist.sample()
                bg_features = bg_features * vae.config.scaling_factor
                image_masks = batch["image_masks"]

                masked_images = batch["images"] * (1 - image_masks).unsqueeze(1)
                masked_features = vae.encode(masked_images.to(weight_dtype)).latent_dist.sample()
                masked_features = masked_features * vae.config.scaling_factor
                segmentation_masks = batch["segmentation_masks"]
                image_masks_256 = F.interpolate(image_masks.unsqueeze(1), size=(256, 256), mode='nearest')
                segmentation_masks = image_masks_256 * segmentation_masks
                glyph = batch['glyphs']
                bg = batch['bgs']
                feature_masks = F.interpolate(image_masks.unsqueeze(1), size=(64, 64), mode='nearest')

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(features)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (features.shape[0], features.shape[1], 1, 1), device=features.device
                    )

                bsz = features.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=features.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(features, noise, timesteps)

                encoder_hidden_states_gp = image_encoder(**glyph).last_hidden_state.to(weight_dtype)
                encoder_hidden_states_bg = image_encoder(**bg).last_hidden_state.to(weight_dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states_bg,encoder_hidden_states_gp],1)
                # encoder_hidden_states = text_encoder(batch["prompts"]).last_hidden_state.to(weight_dtype)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":  # √
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(features, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if accelerator.is_main_process:
                    if (step + 1) % args.vis_interval == 0:
                        scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path,
                                                                  subfolder="scheduler")
                        scheduler.set_timesteps(50)
                        noise = torch.randn((args.vis_num, 4, 64, 64)).to("cuda")
                        input = noise
                        for t in tqdm(scheduler.timesteps):
                            with torch.no_grad():
                                noisy_residual = unet(input, t, encoder_hidden_states=encoder_hidden_states[:args.vis_num],
                                                      masked_feature=masked_features[:8],
                                                      feature_mask=feature_masks[:8],
                                                      segmentation_mask=segmentation_masks[:8],bg_feature = bg_features[:8]
                                                      ).sample
                                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                                input = prev_noisy_sample


                        # decode
                        input = 1 / vae.config.scaling_factor * input
                        images = vae.decode(input.half(), return_dict=False)[0]

                        ## save predicted images
                        width, height = 512, 512
                        new_image = Image.new('RGB', (4 * width, 4 * height))
                        for index, image in enumerate(images.float()):
                            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                            row = index // 4
                            col = index % 4
                            new_image.paste(image, (col * width, row * height))
                        new_image.save(f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_pred_img.png')

                        ## save segmentation masks
                        width, height = 512, 512
                        new_image = Image.new('L', (4 * width, 4 * height))
                        for index, image in enumerate(segmentation_masks[:args.vis_num]):
                            segmap_pil = Image.fromarray(((image != 0) * 255).squeeze().cpu().numpy().astype("uint8"))
                            row = index // 4
                            col = index % 4
                            new_image.paste(segmap_pil, (col * width, row * height))
                        new_image.save(
                            f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_segmentation_mask.png')

                        ## save original images
                        width, height = 512, 512
                        new_image = Image.new('RGB', (4 * width, 4 * height))
                        for index, image in enumerate(batch["images"][:args.vis_num]):
                            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                            # pred_images.append(image)
                            row = index // 4
                            col = index % 4
                            new_image.paste(image, (col * width, row * height))
                        new_image.save(f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_orig_img.png')

                        ## save masked original images
                        width, height = 512, 512
                        new_image = Image.new('RGB', (4 * width, 4 * height))
                        for index, image in enumerate(masked_images[:args.vis_num]):
                            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                            # pred_images.append(image)
                            row = index // 4
                            col = index % 4
                            new_image.paste(image, (col * width, row * height))
                        new_image.save(
                            f'{args.output_dir}/[{epoch}]_{(step + 1) // args.vis_interval}_masked_orig_img.png')

                        print('inference successfully')

                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    masked_feature=masked_features,
                    feature_mask=feature_masks,
                    segmentation_mask=segmentation_masks,
                    bg_feature=bg_features,
                ).sample

                pred_x0 = noise_scheduler.get_x0_from_noise(model_pred, timesteps, noisy_latents)
                with torch.no_grad():
                    input = 1 / vae.config.scaling_factor * pred_x0
                    images = vae.decode(input.half(), return_dict=False)[0]
                    cropped_imgs=[]
                    for index, image in enumerate(images.float()):
                        image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                        image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                        points = batch['points'][index]
                        xmin = min(point[0] for point in points)
                        ymin = min(point[1] for point in points)
                        xmax = max(point[0] for point in points)
                        ymax = max(point[1] for point in points)
                        cropped_img = image.crop((xmin, ymin, xmax, ymax))
                        cropped_img = preprocess(cropped_img)
                        cropped_imgs.append(cropped_img)
                reg_img = torch.cat(cropped_imgs, dim=0).to(accelerator.device)
                outputs = reg(reg_img, mode="validation")
                celoss_inputs = outputs[:3]
                gt_ids, gt_lengths = prepare_label(
                    batch["texts"], charset,accelerator.device)
                reg_loss = MultiLosses(True)(celoss_inputs, gt_ids, gt_lengths)
                resized_charmap = F.interpolate(batch["segmentation_masks"].float(), size=(64, 64),
                                                mode="nearest").long()

                ce_loss = ce_criterion(segmenter(pred_x0.float()), resized_charmap.squeeze(1))

                mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = mse_loss + ce_loss * args.character_aware_loss_lambda + reg_loss*args.reg_loss_lambda

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                    'mse_loss': mse_loss.detach().item(), 'ce_loss': ce_loss.detach().item(),'reg_loss':reg_loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=image_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
