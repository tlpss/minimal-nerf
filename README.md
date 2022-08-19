# minimal-nerf
Implementation of [vanilla NeRF](https://arxiv.org/abs/2003.08934) in Pytorch (using pytorch lightning and weights and biases).


This repo is for learning about NeRFs. If you simply want to train a NeRF on your own data, don't use this repo, use a fast implementation such as [NVIDIA instant-NGP](https://github.com/NVlabs/instant-ngp).



The repo is inspired by the [original repository](https://github.com/bmild/nerf) and the author's [tiny nerf](https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb). W.r.t the original paper, hierarchical sampling is not implemented for now.


Two aspects that I struggled most with in this implementation:
1. the importance of the min/max render distance parameters. Initially these were set to values that were not appropriate for the scene at hand, resulting in all-black images and a constant loss with gradients that became zero.
2. The sigmoid activation for the density estimation can result in zero gradients (as discussed [here](https://github.com/sxyu/pixel-nerf/issues/15#issuecomment-812947300)) and I would see that for some seeds the nerf got stuck, rendering all-black screens. Therefore I changed it to a softplus activation, which made training more stable.
3. (not necessarily directly related to NeRF), but many parts need to come together before you can see whether it works or not, and hence finding what is can be hard.. I therefore at one moment separated my rendering code and compared it on random tensors to the code of [this popular pytorch implementation](https://github.com/yenchenlin/nerf-pytorch), which helped a lot.




## Train the NeRF on the tiny LEGO dataset

- `conda env create -n nerf python=3.9`
- `conda env update -f environment.yaml` # todo
- `python minimal_nerf/train.py --gpus 1`


## Parameter Importance
A hyperparameter sweep to assess the importance of the different parameters of the system on the tiny LEGO dataset can be found [here](https://wandb.ai/tlips/nerf-hackathon/sweeps/849e7nhd?workspace=user-tlips).


### TODOs
- env.yaml file
- adjust default hparams based on sweep
- add dataloader for Colmap data
- hierarchical sampling?
- clean up video generation code
