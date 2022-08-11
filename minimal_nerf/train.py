import inspect
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import  ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from minimal_nerf.nerf import NeRF
from minimal_nerf.datasets.tiny_nerf_dataset import TinyNerfDataset, TinyNerfImageDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import wandb 

def create_pl_trainer(hparams: dict, wandb_logger: WandbLogger) -> Trainer:
    """
    function that creates a pl.Trainer instance from the given global hyperparameters and logger
    and adds some callbacks.
    """

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: hparams[name] for name in valid_kwargs if name in hparams}
    trainer_kwargs.update({"logger": wandb_logger})

    # cf https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.wandb.html

    checkpoint_callback = ModelCheckpoint(monitor="train/loss", mode="min")

    trainer = pl.Trainer(**trainer_kwargs, callbacks=[ checkpoint_callback])
    return trainer

if __name__ == "__main__":
    dataset  = TinyNerfDataset()
    dataloader = DataLoader(dataset,batch_size=4096,shuffle=True)

    val_dataset =  TinyNerfImageDataset(indices=[0,10,20,30])
    val_dataloader = DataLoader(val_dataset,batch_size=1,shuffle=False)

    # create the parser, add module arguments and the system arguments
    parser = ArgumentParser()
    parser = NeRF.add_model_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)


    # get parser arguments and filter the specified arguments
    hparams = vars(parser.parse_args())
    # remove the unused optional items without default, which have None as key
    hparams = {k: v for k, v in hparams.items() if v is not None}
    print(f" argparse arguments ={hparams}")

    # initialize wandb here, this allows for using wandb sweeps.
    # with sweeps, wandb will send hyperparameters to the current agent after the init
    # these can then be found in the 'config'
    # (so wandb params > argparse)
    wandb.init(
        project="nerf-hackathon",
        config=hparams,
    )


    # get (possibly updated by sweep) config parameters
    hparams = wandb.config
    print(f" config after wandb init: {hparams}")

    print("starting training")

    logger = WandbLogger() # takes values from current wand session ^
    pl.seed_everything(2023)
    nerf = NeRF(**hparams)
    trainer = create_pl_trainer(hparams, logger)
    trainer.fit(nerf,dataloader,val_dataloader)