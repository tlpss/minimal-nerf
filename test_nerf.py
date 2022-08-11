import numpy as np 
import matplotlib.pyplot

from tiny_nerf_dataset import TinyNerfDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from nerf import NeRF
if __name__ == "__main__":

    logger = WandbLogger(project="test")
    dataset  = TinyNerfDataset()
    dataloader = DataLoader(dataset,batch_size= 4096,shuffle=False)
    nerf = NeRF()
    logger.watch(nerf,log_freq=1)
    trainer = pl.Trainer(max_epochs=10,gpus=1,logger=logger)
    trainer.fit(nerf,dataloader)

    # batch = next(iter(dataloader))
    # rays_o, rays_d, pixels = batch 
    # optimizer = torch.optim.Adam(nerf.parameters(),lr=3e-4)
    # for _ in range(10):
    #    loss = nerf.training_step(batch,0)
    #    print(loss)
    #    loss.backward()
    #    print(nerf.nerf.rgb_head.bias.grad)
    #    print(torch.max(nerf.nerf.encoding[2].bias.grad))
    #    optimizer.step()
    #    optimizer.zero_grad()



