from PIL import Image, ImageDraw
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
import math
import cv2
from torchvision import transforms
import re

class DiffData(Dataset):

    def __init__(self, root_dir,mask_all_ratio):
        self.root_dir = root_dir
        self.img_path = os.path.join(self.root_dir,'img')
        self.img_list = os.listdir(self.img_path)
        self.glyph_path = os.path.join(self.root_dir, 'glyph')
        self.ann_path = os.path.join(self.root_dir, 'ann')
        self.seg_path = os.path.join(self.root_dir,'seg')
        self.bg_path = os.path.join(self.root_dir,'bg')
        self.mask_all_ratio = mask_all_ratio
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def get_mask(self,ocrs,w,h):
        # if random.random() <= self.mask_all_ratio:
        #     image_mask = Image.new('L', (512, 512), 1)
        #     return image_mask

        image_mask = Image.new('L', (512, 512), 0)
        draw_image_mask = ImageDraw.ImageDraw(image_mask)
        scale_x = 512.0 / w
        scale_y = 512.0 / h
        for ocr in ocrs:
            coords_text = ocr.strip().split(',')
            box = list(map(int, coords_text[:8]))
            text = coords_text[8:]
            result = ''.join(text)
            result = re.sub(r'[^a-zA-Z0-9]', '', result)
            box = [int(i * scale_x) if idx % 2 == 0 else int(i * scale_y) for idx, i in enumerate(box)]
            points = [(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])]
            if random.random() < self.mask_all_ratio:  # each box is masked with 70% probability
                draw_image_mask.polygon(points, fill=1)
        return image_mask,result,points



    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        base_name = img_name.split('.')[0]
        ann_name = base_name + '.txt'
        seg_name = base_name+'.npy'
        with open(os.path.join(self.ann_path, ann_name), 'r') as f:
            ocrs = f.readlines()
        image = Image.open(os.path.join(self.img_path,img_name)).convert("RGB")
        width, height = image.size
        image = image.resize((512,512))
        image_mask,text,points = self.get_mask(ocrs,width,height)
        image_mask_np = np.array(image_mask)
        image_mask_tensor = torch.from_numpy(image_mask_np)
        segmentation_mask = np.load(os.path.join(self.seg_path,seg_name))
        random_value = random.random()
        if random_value < 0.8:
            pass
        elif random_value < 0.85:
            kernel = np.ones((3, 3), dtype=np.uint8)
            segmentation_mask = cv2.dilate(segmentation_mask.astype(np.uint8), kernel, iterations=1)
        elif random_value < 0.9:
            kernel = np.ones((3, 3), dtype=np.uint8)
            segmentation_mask = cv2.erode(segmentation_mask.astype(np.uint8), kernel, iterations=1)
        elif random_value < 0.95:
            kernel = np.ones((3, 3), dtype=np.uint8)
            segmentation_mask = cv2.dilate(segmentation_mask.astype(np.uint8), kernel, iterations=2)
        else :
            kernel = np.ones((3, 3), dtype=np.uint8)
            segmentation_mask = cv2.erode(segmentation_mask.astype(np.uint8), kernel, iterations=2)

        image = self.train_transforms(image).sub_(0.5).div_(0.5)
        glyph = Image.open(os.path.join(self.glyph_path, img_name)).convert("RGB")
        bg = Image.open(os.path.join(self.bg_path, img_name)).convert("RGB")
        bg_conv = bg.resize((512, 512))
        bg_conv = self.train_transforms(bg_conv).sub_(0.5).div_(0.5)
        return image,image_mask_tensor,segmentation_mask,glyph,bg,text,points,bg_conv

    def __len__(self):
        return len(self.img_list)