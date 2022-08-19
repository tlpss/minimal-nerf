# minimal-nerf
Implementation of vanilla NeRF in Pytorch (using pytorch lightning and weights and biases).


This repo is for learning about NeRFs. If you simply want to train a NeRF on your own data, don't use this repo, use a fast implementation such as [NVIDIA instant-NGP](https://github.com/NVlabs/instant-ngp).



The repo is inspired by the [original repository](https://github.com/bmild/nerf) and the author's [tiny nerf](https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb). W.r.t the original paper, hierarchical sampling is not implemented for now.

A hyperparameter sweep to assess the importance of the different parameters of the system on the tiny LEGO dataset can be found [here](https://wandb.ai/tlips/nerf-hackathon/sweeps/849e7nhd?workspace=user-tlips).


TODOs:
- env.yaml file
- adjust default hparams based on sweep
- add dataloader for Colmap data
- hierarchical sampling?
- clean up video generation code

## Train the NeRF on the tiny LEGO dataset

- `conda env create -n nerf python=3.9`
- `conda env update -f environment.yaml` # todo
- `python minimal_nerf/train.py --gpus 1`



