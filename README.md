# ICL-SAM: Synergizing In-context Learning Model and SAM in Medical Image Segmentation

## 📄 Introduction

This repo contains the code for our paper:  
**"ICL-SAM: Synergizing In-context Learning Model and SAM in Medical Image Segmentation"**, accepted at **MIDL 2024 (Oral)**.  
📖 [Read the paper](https://openreview.net/pdf/4d1b81a2fc459e43945becaa6df59c4f5b744fc9.pdf)

ICL-SAM integrates:
- 🧩 [UniverSeg](https://github.com/JJGO/UniverSeg/tree/main) — for in-context few-shot learning  
- 🎯 [SAM](https://github.com/facebookresearch/segment-anything) — for powerful interactive segmentation

It enables flexible and generalizable medical image segmentation with just a few annotated examples — **no retraining and interactive needed!!**.

---

## ⚠️ Note on Use

ICL-SAM works best when **SAM performs reasonably well on its own**.  
In challenging cases where SAM struggles, adding ICL may **not improve results**.

**Key insight**:  
- **ICL** reduces reliance on manual prompts  
- **SAM** captures fine details interactively  
➡️ They offer **complementary strengths**.

---



<div align="center">
  <img src="figs/framework.png"/ width="97%"> <br>
</div>

<div align="center">
  <img src="figs/curve.png"/ width="97%"> <br>
</div>

We evaluated our method in 3 different types of datasets, and the results showed that our method can greatly improve previous in-context learning model especially when the support set is small.  

## Requirements
Use the following commands to create and activate the necessary environment:

```bash
conda create -n ICLSAM python=3.8
conda activate ICLSAM

pip install -r requirements.txt
```

ICL-SAM's code requires pytorch>=1.7 and torchvision>=0.8. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies.

## Download Checkpoints
Before getting started, download the checkpoints for SAM or MedSAM:

- [SAM ViT-b Checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [MedSAM Checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link)  

## Getting Started
Just run the `ICL-SAM_example.ipynb` notebook and follow the step-by-step instructions. The notebook will guide you through the process of using our model. Make sure change the checkpoint path before running the demo.

## Citation
If you find our work useful, please consider citing:

```
@inproceedings{hu2024synergizing,
  title={Synergizing In-context Learning Model and SAM in Medical Image Segmentation},
  author={Hu, Jiesi and Shang, Yang and Yang, Yanwu and Xutao, Guo and Peng, Hanyang and Ma, Ting},
  booktitle={Medical Imaging with Deep Learning},
  year={2024}
}
```

## Acknowledgements
This repository benefits from the excellent work provided by [UniverSeg](https://github.com/JJGO/UniverSeg/tree/main) and [Personalize-SAM](https://github.com/ZrrSkywalker/Personalize-SAM). We extend our gratitude for their significant contributions to the field.



