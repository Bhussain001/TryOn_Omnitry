<p align="center">

  <h2 align="center">OmniTry: Virtual Try-On Anything without Masks</h2>
  <p align="center">
    <a href="https://scholar.google.com.hk/citations?user=mZwJLeUAAAAJ&hl=zh-CN"><strong>Yutong Feng</strong></a>
    ·
    <a href=""><strong>Linlin Zhang</strong></a>
    ·
    <a href=""><strong>Hengyuan Cao</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=LxiMyjQAAAAJ"><strong>Yiming Chen</strong></a>
    ·
    <a href=""><strong>Xiaoduan Feng</strong></a>
    ·
    <a href=""><strong>Jian Cao</strong></a>
    ·
    <a href=""><strong>Yuxiong Wu</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=6hTbqDEAAAAJ&hl=zh-CN"><strong>Bin Wang</strong></a>
    <br>
    <b>Kunbyte AI &nbsp; | &nbsp;  Zhejiang University </b>
    <br>
    <br>
        <a href="http://arxiv.org/abs/2508.13632"><img src='https://img.shields.io/badge/arXiv-OmniTry-red' alt='Paper PDF'></a>
        <a href="https://omnitry.github.io/"><img src='https://img.shields.io/badge/project page-OmniTry-green'></a>
        <a href='https://huggingface.co/Kunbyte/OmniTry'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow'></a>
        <a href='https://huggingface.co/spaces/Kunbyte/OmniTry'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
        <a href='https://huggingface.co/datasets/Kunbyte/OmniTry-Bench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-orange'></a>
    <br>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/teaser.png">
    </td>
    </tr>
  </table>

## News
* **[2025.08.20]** 🎉🎉🎉 We release the model weights, inference demo and evaluation benchmark of OmniTry! To experience our advanced version and other related features, please visit our product website [k-fashionshop](https://marketing.k-fashionshop.com/home) (in Chinese) or [visboom](https://www.visboom.com/) (in English).
* **[2025.08.29]** 🛠️🛠️🛠️ We release the data pre-processing scripts for OmniTry! Now you can process your own dataset following [data_preprocess](./data_preprocess/README.MD).

## Get Started

**Noted**: Currently, OmniTry requires at least **28GB** of VRAM for inference under torch.bfloat16. We will continue work to decrease memory requirements.

### Download Checkpoints
1. Create the checkpoint directory: `mkdir checkpoints`

2. Download the [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) into `checkpoints/FLUX.1-Fill-dev`

3. Download the [LoRA of OmniTry](https://huggingface.co/Kunbyte/OmniTry) into `checkpoints/omnitry_v1_unified.safetensors`. You can also download the `omnitry_v1_clothes.safetensors` that specifically finetuned on the clothe data only.

### Environment Prepartion
Install the environment with `conda`
```bash
conda env create -f environment.yml
conda activate omnitry
```
or `pip`:
```bash
pip install -r requirements.txt
```

(Optional) We recommend to install the [flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main) to accelerate the inference process:
```bash
pip install flash-attn==2.6.3
```

### Usage
For running the gradio demo:
```bash
python gradio_demo.py
```

To change different versions of checkpoints for OmniTry, replace the `lora_path` in `configs/omnitry_v1_unified.yaml`.

## OmniTry-Bench
We present a unified evaluation benchmark for OmniTry. Please refer to the [OmniTry-Bench](./omnitry_bench/README.MD).


## Acknowledgements
This project is developped on the [diffusers](https://github.com/huggingface/diffusers) and [FLUX](https://github.com/black-forest-labs/flux). We appreciate the contributors for their awesome works.


## Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@article{feng2025omnitry,
  title={OmniTry: Virtual Try-On Anything without Masks},
  author={Feng, Yutong and Zhang, Linlin and Cao, Hengyuan and Chen, Yiming and Feng, Xiaoduan and Cao, Jian and Wu, Yuxiong and Wang, Bin},
  journal={arXiv preprint arXiv:2508.13632},
  year={2025}
}
```