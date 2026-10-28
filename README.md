# Mirai: Autoregressive Visual Generation Needs Foresight

<div align="center">

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_Mirai_Autoregressive_Visual_Generation_Needs_Foresight_CVPR_2026_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2601.14671-b31b1b.svg)](https://arxiv.org/abs/2601.14671)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://y0uroy.github.io/Mirai/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

<p align="center">
  <img src="assets/sample.png" width="800">
</p>

This is the official PyTorch implementation of the paper **"Mirai: Autoregressive Visual Generation Needs Foresight"** (CVPR 2026).

> Our study highlights that visual autoregressive models need foresight.

<!-- > **Abstract:** *Autoregressive (AR) visual generators model images as sequences of discrete tokens... We propose Mirai, a general framework that injects future information into AR training...* -->

<!-- ## 📅 News
* **[2025-12-14]** Codes are released!
* **[2025-XX-XX]** Paper is available on arXiv. -->


## 🛠️ Installation

```bash
git clone https://github.com/y0urOy/Mirai.git
cd Mirai
conda create -n mirai python=3.9
conda activate mirai
pip install -r requirements.txt
```
## ⬇️ Download

```bash
mkdir pretrained_models
cd pretrained_models
```
Download [vq_ds16_c2i.pt](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt) 



## 🧠 Dataset and preprocessing

We mainly on ImageNet 256x256:

```bash
python autoregressive/train/extract_codes.py \
  --vq-ckpt pretrained_models/vq_ds16_c2i.pt \
  --data-path ImageNet/train \
  --code-path imagenet_code_c2i_flip_ten_crop \
  --ten-crop \
  --crop-range 1.1 \
  --image-size 256
```

## 🚀 Training

To train Mirai-I on ImageNet 256x256:

```bash
torchrun --nnodes=1 --nproc_per_node=8    --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 \  
  autoregressive/train/train_c2i_Mirai_I.py \
  --gpt-model="GPT-B" \
  --image-size=256 \
  --downsample-size=16 \
  --resolution 256 \
  --global-batch-size=256 \
  --lr=1e-4 \
  --epochs=300 \
  --ema \
  --results-dir="" \
  --cloud-save-path="results_cloud" \
  --dataset="imagenet_json" \
  --code-path="imagenet_code_c2i_flip_ten_crop" \
  --raw-image-path="/ImageNet/train" \
  --json-path "imagenet_code_c2i_flip_ten_crop/imagenet_256_manifest.json" \
  --teacher-depth=8 \
  --student-depth=8 \
  --enc-type="dinov2-vit-b" \
  --report-to-wandb \
  --proj-coeffs 2 \
  --warmup-epochs 0 \
  --num-heads=1 \
  --ckpt-every 100000
```

To train Mirai-E on ImageNet 256x256:

```bash
torchrun --nnodes=1 --nproc_per_node=8  --node_rank=0 --master_addr=127.0.0.1 --master_port=12345 \   
  autoregressive/train/train_c2i_Mirai_E.py \
  --gpt-model="GPT-B" \
  --image-size=256 \
  --downsample-size=16 \
  --global-batch-size=256 \
  --lr=1e-4 \
  --epochs=80 \
  --ema \
  --results-dir="results" \
  --cloud-save-path="results_cloud" \
  --dataset="imagenet_code" \
  --code-path="imagenet_code_c2i_flip_ten_crop" \
  --teacher-depth=8 \
  --student-depth=8 \
  --report-to-wandb \
  --proj-coeffs 2 2 2 \
  --use-prev-iter-ema \
  --warmup-epochs 15 \
  --num-heads=3 \
  --ckpt-every 100000
```

## ⚡ Inference / Sampling

<!-- Download our pretrained models from HuggingFace and run: -->

```bash
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_port=12345 \   
  autoregressive/sample/sample_c2i_ddp.py \
  --vq-ckpt pretrained_models/vq_ds16_c2i.pt \
  --gpt-ckpt results/001-GPT-B/checkpoints/0400000.pt \
  --gpt-model GPT-B \
  --image-size 256 \
  --image-size-eval 256 \
  --cfg-scale 2 \
  --ema
```

## 📊 Evaluation & Metrics
Before evaluation, please refer [evaluation readme](evaluations/README.md) to install required packages. 
<!-- Download our pretrained models from HuggingFace and run: -->

```bash
python3 evaluations/evaluator.py \
  evaluations/c2i/VIRTUAL_imagenet256_labeled.npz \
  samples/GPT-B-0400000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0.npz
```

## 🎓 Citation
If you find our work helpful, please cite:

```bibtex
@InProceedings{Yu_2026_CVPR,
    author    = {Yu, Yonghao and Huang, Lang and Wang, Zerun and Li, Runyi and Yamasaki, Toshihiko},
    title     = {Mirai: Autoregressive Visual Generation Needs Foresight},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {30511-30520}
}

```

## Acknowledgements
This codebase is built upon [LlamaGen](https://github.com/FoundationVision/LlamaGen). We thank the authors for their open-source contribution.

## Project website

The project page is published at [https://y0uroy.github.io/Mirai/](https://y0uroy.github.io/Mirai/).
Its source is in [website/](website/); GitHub Actions publishes that directory at the same address.
The code and website histories are preserved in this repository.

## License

See [LICENSE](LICENSE) and the license headers of vendored files. The website retains its
[Creative Commons Attribution-ShareAlike 4.0 attribution](website/README.md#website-license).
