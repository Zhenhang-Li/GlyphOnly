# First Creating Backgrounds Then Rendering Texts: A New Paradigm for Visual Text Blending

<a href='https://arxiv.org/abs/2410.10168'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/Zhenhang-Li/GlyphOnly/'><img src='https://img.shields.io/badge/Code-Github-green'></a>

## TODOs
- [x] Release SynthGlyph10K and Better_Background
- [ ] Release inference code and checkpoints
- [ ] Release training code
- [x] Release training dataset and test dataset
- [ ] I will open-source it as soon as possible. If you need anything, feel free to contact me at lizhenhang0506@gmail.com. I will provide the unorganized code and datasets.

## Datasets
A better background image for generating scene text images is available at [Google Drive](https://drive.google.com/drive/folders/1WD3tHFXW3Rls6OQIIs8WcngFU-ZPN49s?usp=sharing).
SynthGlyph10K is available at [Google Drive](https://drive.google.com/file/d/1jDmyplx30kfitUNdSNAkoZqkE2YfIKid/view?usp=sharing).
Instance categories suitable for text generation on surfaces can be obtained at [Google Drive](https://drive.google.com/file/d/1ULTkLcfVUtVgjflE2LAeJ9pFg_N4QxG-/view?usp=sharing).
Train dataset is available at [Google Drive](https://drive.google.com/file/d/10rs0cxSy9KkJ0eliSy16fzCN6Paw1fTm/view?usp=sharing).
Test dataset is available at  [Google Drive](https://drive.google.com/file/d/1V-ikGrtNBpLkTbN8bxM5Efp_qDTVMRBT/view?usp=sharing).
## Installation
Clone this repo: 
```
git clone https://github.com/Zhenhang-Li/GlyphOnly.git
cd GlyphOnly
```

Build up a new environment and install packages as follows:
```
conda create -n glyphonly python=3.8
conda activate glyphonly
pip install -r requirements.txt
```
## Inference

## Training

## Evaluation

## Citation
```
@incollection{li2024first,
  title={First Creating Backgrounds Then Rendering Texts: A New Paradigm for Visual Text Blending},
  author={Li, Zhenhang and Shu, Yan and Zeng, Weichao and Yang, Dongbao and Zhou, Yu},
  booktitle={ECAI 2024},
  pages={346--353},
  year={2024},
  publisher={IOS Press}
}
```
