# ADAPT++

 <img src="fig3.png" width="650"> 

This repository is an official implementation of ADAPT++

Created by yufan shu

## Introduction

We propose an end-to-end transformer-based architecture, we propose An abnormal behavior dataset, and the driving behavior understanding and retrieval of video data are realized

This repository contains the training and testing of the proposed framework in paper.

## Note
This reposity will be updated soon, including:
- [] Uploading the **[Preprocessed Data](#dataset-preparation)** of dataset.
- [] Uploading the **Raw Data** of IAAD.
- [] Uploading the **Visualization Codes** of raw data and results.
- [] Updating the **Experiment Codes** to make it easier to get up with.
- [x] Uploading the **[Conda Environments](#1-installation-as-conda)** of ADAPT++.



## Getting Started


### 1. Installation as Conda

Create conda environment:
```
conda create --name adapt++ python=3.8
conda activate adapt++
```

install torch
```
pip install torch==1.13.1+cu116 torchaudio==0.13.1+cu116 torchvision==0.14.1+cu116 -f 
```

install apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -r requirements.txt
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
(I highly recommend going to the apex repository in person to see if the install commands are updated, I've been bothered by this issue)
cd ..
rm -rf apex
```

install requirements
```
apt-get install mpich
sudo apt-get update
sudo apt-get install libopenmpi-dev
pip install mpi4py
pip install -r requirements.txt
```





## Acknowledgments

Our code is built on top of open-source GitHub repositories. 
We thank all the authors who made their code public, which tremendously accelerates our project progress. 
If you find these works helpful, please consider citing them as well.

[jxbbb/ADAPT](https://github.com/jxbbb/ADAPT)

[Microsoft/SwinBERT](https://github.com/microsoft/SwinBERT) 

[JinkyuKimUCB/BDD-X-dataset](https://github.com/JinkyuKimUCB/BDD-X-dataset)

[huggingface/transformers](https://github.com/huggingface/transformers) 

[Microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)

[Nvidia/Apex](https://github.com/NVIDIA/apex)

[FAIR/FairScale](https://github.com/facebookresearch/fairscale)

