# CODI

This is the official implementation of the paper: [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074)

![codi](imgs/codi_method_v4.png)

## Setup

Clone repo:
```
git clone git@github.com:zhenyi4/CODI.git
cd CODI
```

Setup environment:
```
conda create --name codi python=3.12
conda activate codi
pip install -r requirements.txt
```

## Run the results

The model weights are available at https://huggingface.co/zen-E.

Run the accuracy result on GSM8k.
```
bash script/test_gpt2.sh # or script/test_llama1b.sh
```

Interpret the latent thoughts on GSM8k (Section 5 in the paper).
```
bash script/probe_latent_token.sh
```

## Training
```
bash script/train_gpt2.sh # or script/train_llama1b.sh
```

## Key Arguments
use_prj: Whether use a projection layer for the last layer hidden state.

prj_dim: The dimension of the hidden state of the projection layer.

prj_no_ln: Whether the projection layer is not followed by a LayerNorm layer.

mse_loss_div_std: Whether divide the distillation loss via the standard deviation of the teacher's hidden states.

mse_loss_type: The type of loss used for distillation (e.g. l1, l2, smoothl1).

mse_loss_factor: The multiplier that scales the distillation loss in the total loss calculation.

ref_loss_factor: The multiplier that scales the teacher's cross entropy loss in the total loss calculation.

num_latent: The number of latent thoughts used for training.

inf_latent_iterations: The number of latent thoughts used for inference.

include_last_cot: Include the last CoT step in the training data.

fix_attn_mask: An argument that fixs a bug. Can leave it as False.

max_token_num: Training data that have more tokens than this value are discarded.

## Citation
If you use this code base in your research, please cite our paper with the following BibTex entry:
```bibtex
@article{shen2025codi,
      title={CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation}, 
      author={Zhenyi Shen and Hanqi Yan and Linhai Zhang and Zhanghao Hu and Yali Du and Yulan He},
      year={2025},
      journal={arXiv preprint arxiv:2502.21074},
}
```
