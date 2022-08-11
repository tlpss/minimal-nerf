from dis import dis
from random import shuffle
from turtle import distance
from typing import Tuple
from unittest import TestResult
import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
import torch.random 
import wandb 
from pytorch_lightning.loggers import WandbLogger

class NeRFNetwork(nn.Module):
    def __init__(self, depth=2, width=256, skips=[]):
        super().__init__()
        self.depth = depth
        self.width = width 
        self.skips = skips

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.feature_layers = nn.ModuleList([nn.Linear(3,self.width)])
        for i in range(self.depth-1):
            depth = i +1
            if depth in self.skips:
                self.feature_layers.append(nn.Linear(self.width+3,self.width))
            else:
                self.feature_layers.append(nn.Linear(self.width,self.width))

        self.density_head = nn.Linear(self.width,1)
        self.feature_head = nn.Linear(self.width,self.width)
        self.rgb_layer = nn.Linear(self.width +3, self.width//2)
        self.rgb_head = nn.Linear(self.width//2,3)

    def forward(self,x):

        position, view_direction = torch.split(x,3,dim=-1)
        x = position
        for i,layer in enumerate(self.feature_layers):
            if i in self.skips:
                x = torch.cat([x,position],dim=-1)
            x = layer(x)
            x = self.relu(x)

        density = self.density_head(x)
        density = self.relu(density) # constrain to 0- Inf.

        features = self.feature_head(x)
        features = self.relu(features)

        x = torch.cat([features, view_direction],dim=-1)
        x = self.rgb_layer(x)
        x = self.relu(x)
        rgb = self.rgb_head(x)
        rgb = self.sigmoid(rgb) # constrain to 0-1


        return rgb, density
   
class NeRF(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()


        #TODO: get these from colmap?
        self.min_render_distance = 2.0
        self.max_render_distance = 6.0

        self.n_coarse_samples = 64
        self.nerf = NeRFNetwork()

    def forward(self, x):
        return self.nerf(x)

    
    def render_view(self,camera_pose, height, width,focal_distance):
        rays_origin, rays_direction = get_image_rays(camera_pose.cpu(),height,width,focal_distance, device=self.device)
        rendered_rgbs = self.render_rays(rays_origin.flatten(end_dim=-2),rays_direction.flatten(end_dim=-2),self.n_coarse_samples)
        rendered_img = rendered_rgbs.reshape(rays_origin.shape)
        return rendered_img


    def render_rays(self,ray_origins, ray_directions, n_coarse_samples):

        network_query_points, z_sample_points = self._compute_query_points(ray_origins,ray_directions,n_coarse_samples)
        network_query_inputs = torch.cat([network_query_points, ray_directions.unsqueeze(1).expand(-1,n_coarse_samples,-1)],dim=-1)
        queried_rgbs, queried_densities = self.forward(network_query_inputs)

        rendered_rgbs = self._volume_render_rays(z_sample_points,queried_rgbs, queried_densities)

        return rendered_rgbs 

    def _compute_query_points(self,ray_origins:torch.Tensor, ray_directions: torch.Tensor, n_coarse_samples:int) -> torch.Tensor:
        """

        Args:
            ray_origins (_type_): N_rays x 3 
            ray_directions (_type_): N_rays x 3
            n_coarse_samples (_type_): 

        Returns:
            query point: n_rays x n_coarse_samples x 3
            z_distances: n_rays x n_coarse_samples x 1 
        """
        bins = torch.linspace(self.min_render_distance, self.max_render_distance,steps =n_coarse_samples +1,device=self.device)
        bin_width = (self.max_render_distance-self.min_render_distance)/n_coarse_samples
        uniform_bin_samples = torch.rand(ray_origins.shape[0],n_coarse_samples,device=self.device) 
        uniform_bin_samples *= bin_width
        z_sample_points = bins[:-1].unsqueeze(0).unsqueeze(-1).expand(ray_directions.shape[0],-1,-1) #+ uniform_bin_samples

        
        expanded_ray_directions = ray_directions.unsqueeze(1).expand(-1,n_coarse_samples,-1) # (N_rays x N_samples x 3)
        query_points = ray_origins.unsqueeze(1) + expanded_ray_directions * z_sample_points
        return query_points, z_sample_points

    def _volume_render_rays(self, z_sample_points, queried_rgbs, queried_densities):
        """_summary_

        Args:
            z_sample_points (_type_): N_rays x N_samples x 1
            queried_rgbs (_type_): N_rays x N_samples x 3
            queried_densities (_type_): N_rays x N_samples x 1

        Returns:
            _type_: _description_
        """
        queried_densities = queried_densities #TODO: check if random noise helps? +  torch.randn_like(queried_densities) * 0.01
        queried_densities = self.nerf.relu(queried_densities)
        distances = self.compute_adjacent_distances(z_sample_points)
        neg_weighted_densities = - distances * queried_densities + 1e-10
        zeros_shape = (neg_weighted_densities.shape[0],1,1)
        transmittances = torch.cat([torch.zeros(zeros_shape,device=self.device),neg_weighted_densities],dim=-2)[:,:-1,:]
        accumulated_transmittances = torch.exp(torch.cumsum(transmittances,dim=-2))
        alpha = (1- torch.exp(neg_weighted_densities))
        weights = alpha * accumulated_transmittances
        rgbs = torch.sum(weights * queried_rgbs, dim = -2)

        if self.logger:
            wandb.log({
                "debug/queried_densities":wandb.Histogram(queried_densities.detach().cpu().numpy()),
                "debug/queried_rgbs": wandb.Histogram(queried_rgbs.detach().cpu().numpy()),
                "debug/alpha": wandb.Histogram(alpha.detach().cpu().numpy()),
                "debug/accum_transmittances": wandb.Histogram(accumulated_transmittances.detach().cpu().numpy()),
                "debug/distances": wandb.Histogram(distances.detach().cpu().numpy()),
                "debug/rendered_rgbs": wandb.Histogram(rgbs.detach().cpu().numpy()),

        

                })
        return rgbs
        

    def compute_adjacent_distances(self,z_sample_points: torch.Tensor) -> torch.Tensor:
        """computes adjacent value distances for each sample. Adds epsilon distance for the last sample.

        Args:
            z_distances (_type_): N_rays x N_samples x 1 tensor

        Returns:
            distances N_rays x N_samples x 1 tensor
        """
        z_sample_points = z_sample_points.squeeze(-1)
        distances = z_sample_points[:, 1:] - z_sample_points[:,:-1]
        eps_distances_for_final_sample = torch.ones((z_sample_points.shape[0],1),device=self.device)*1e10
        distances = torch.cat([distances,eps_distances_for_final_sample], dim =-1)
        return distances.unsqueeze(-1)

    def training_step(self,batch,batch_idx):
        ray_origins, ray_directions,rgb_targets = batch

        rgb_values = self.render_rays(ray_origins, ray_directions,self.n_coarse_samples)

        loss = self.mse(rgb_values,rgb_targets)
        psnr = -10.0 * torch.log(loss) / torch.log(torch.tensor([10.0],device=self.device))
        self.log("train/loss",loss)
        self.log("train/psnr", psnr)
        return loss

    def validation_step(self, batch, batch_idx):
        image, pose, focal_length = batch
        image = image[0]
        pose = pose[0]
        focal_length = focal_length.item()
        with torch.no_grad():
            rendered_image = self.render_view(pose,image.shape[0],image.shape[1],focal_length)
            rendered_image = rendered_image.cpu().numpy()

        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(key=f"train_{batch_idx}", images =[rendered_image], caption=[f"train_{batch_idx}"])



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=4e-4)


def get_image_rays(camera_pose: torch.Tensor, H:int, W:int, focal_distance:float, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """    Get the rays for each pixel in an image, assuming a perfect pinhole camera.


    Args:
        camera_pose (torch.Tensor): 4x4 homogeneous matrix describing camera pose in world
        height (int): _description_
        width (int): _description_
        focal_distance (float): focal distance measured in pixels.

    Returns:
        ray_origins: (torch.Tensor): HxWx3  origin positions
        ray_direction (torch.Tensor): HxWx3 normalized direction vectors
    """

    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    dirs = torch.stack([(i-W//2)/focal_distance, -(j-H//2)/focal_distance, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * camera_pose[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # make sure all rays are normalized!
    direction_norm = torch.linalg.norm(rays_d, dim=-1) 
    rays_d = rays_d / direction_norm.unsqueeze(-1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = camera_pose[:3,-1].expand(rays_d.shape)
    return rays_o.to(device), rays_d.to(device)




def test_rendering():
    from run_nerf import raw2outputs
    n = 2
    samples = 3
    z_vals = torch.rand(n,samples)
    network_output = torch.rand(n,samples,4)
    rgb = network_output[...,:3]
    density = network_output[...,3].unsqueeze(-1)

    rays_d = torch.rand(n,3) 
    rays_d = rays_d / torch.linalg.norm(rays_d, dim = -1).unsqueeze(-1)

    rgb_theirs,*_ = raw2outputs(network_output,z_vals,rays_d)

    nerf = NeRF()
    rgb_mine = nerf._volume_render_rays(z_vals,rgb,density)
    print("rendering difference")
    print(rgb_mine-rgb_theirs)

if __name__ == "__main__":
    # nerf = NeRF()
    #nerf.render_rays(np.zeros((3,3)),None,10)

    import numpy as np 
    import matplotlib.pyplot

    from tiny_nerf_dataset import TinyNerfDataset, TinyNerfImageDataset
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    import torch
    from nerf import NeRF

    logger = WandbLogger(project="test",mode="online")
    dataset  = TinyNerfDataset()
    dataloader = DataLoader(dataset,batch_size=4096,shuffle=True)

    val_dataset =  TinyNerfImageDataset(indices=[0,10,20,30])
    val_dataloader = DataLoader(val_dataset,batch_size=1,shuffle=False)


    nerf = NeRF()
    logger.watch(nerf,log_freq=1)
    trainer = pl.Trainer(max_epochs=50,gpus=1,logger=logger,log_every_n_steps=1)
    trainer.fit(nerf,dataloader,val_dataloader)

    # batch = next(iter(dataloader))
    # rays_o, rays_d, pixels = batch 
    # optimizer = torch.optim.Adam(nerf.parameters(),lr=3e-4)
    # for _ in range(10):
    #    loss = nerf.training_step(batch,0)
    #    print(loss)
    #    loss.backward()
    #    print(nerf.nerf.rgb_head.bias.grad)
    #    optimizer.step()
    #    optimizer.zero_grad()


    #test_rendering()