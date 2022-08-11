from dis import dis
from typing import Tuple
import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
import torch.random 

class NeRFNetwork(nn.Module):
    def __init__(self, depth=8, width=256, skips=[4]):
        super().__init__()
        self.depth = depth
        self.width = width 
        self.skips = skips

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.encoding = nn.ModuleList([nn.Linear(3,self.width)])
        for i in range(self.depth-1):
            depth = i +1
            if depth in self.skips:
                self.encoding.append(nn.Linear(self.width+3,self.width))
            else:
                self.encoding.append(nn.Linear(self.width,self.width))

        self.density_head = nn.Linear(self.width,1)
        self.encoding_head = nn.Linear(self.width,128)
        self.rgb_head = nn.Linear(128+3,3)

    def forward(self,x):

        position, view_direction = torch.split(x,3,dim=-1)
        x = position
        for i,layer in enumerate(self.encoding):
            if i in self.skips:
                x = torch.cat([x,position],dim=-1)
            x = layer(x)
            x = self.relu(x)

        density = self.density_head(x)
        density = self.relu(density) # constrain to 0- Inf.

        features = self.encoding_head(x)
        features = self.relu(features)

        x = torch.cat([features, view_direction],dim=-1)
        rgb = self.rgb_head(x)
        rgb = self.sigmoid(rgb) # constrain to 0-1


        return rgb, density
   
class NeRF(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()


        #todo: get these from colmap?
        self.min_render_distance = 0.0
        self.max_render_distance = 1.0

        self.n_coarse_samples = 64
        self.nerf = NeRFNetwork()

    def forward(self, x):
        return self.nerf(x)

    
    def render_view(self,camera_pose, height, width,intrinsics_matrix, n_coarse_samples):
        pass

    def render_rays(self,ray_origins, ray_directions, n_coarse_samples):

        network_query_points = self._compute_query_points(ray_origins,ray_directions,n_coarse_samples)
        network_query_inputs = torch.cat([network_query_points, ray_directions.unsqueeze(1).expand(-1,n_coarse_samples,-1)],dim=-1)
        queried_rgbs, queried_densities = self.forward(network_query_inputs)

        rendered_rgbs = self._volume_render_rays(network_query_points,queried_rgbs, queried_densities)

        return rendered_rgbs 

    def _compute_query_points(self,ray_origins:torch.Tensor, ray_directions: torch.Tensor, n_coarse_samples:int) -> torch.Tensor:
        """

        Args:
            ray_origins (_type_): N_rays x 3 
            ray_directions (_type_): N_rays x 3
            n_coarse_samples (_type_): 

        Returns:
            n_rays x n_coarse_samples x 3
        """
        bins = torch.linspace(self.min_render_distance, self.max_render_distance,steps =n_coarse_samples +1,device=self.device)
        bin_width = (self.max_render_distance-self.min_render_distance)/n_coarse_samples
        uniform_bin_samples = torch.rand(ray_origins.shape[0],n_coarse_samples,device=self.device) 
        uniform_bin_samples *= bin_width
        samples = bins[:-1] + uniform_bin_samples

        
        expanded_ray_directions = ray_directions.unsqueeze(1).expand(-1,n_coarse_samples,-1) # (N_rays x N_samples x 3)
        query_points = ray_origins.unsqueeze(1) + expanded_ray_directions * samples.unsqueeze(2) # autocasted to ^
        return query_points

    def _volume_render_rays(self, queried_points, queried_rgbs, queried_densities):
        
        distances = self.compute_adjacent_distances(queried_points)
        neg_weighted_densities = - distances * queried_densities
        accumulated_transmittances = torch.exp(torch.cumsum(neg_weighted_densities,dim=-1))

        weights = (1- torch.exp(-distances * queried_densities)) * accumulated_transmittances

        rgbs = torch.sum(weights * queried_rgbs, dim = -2)
        return rgbs
        

    def compute_adjacent_distances(self,query_points: torch.Tensor) -> torch.Tensor:
        """computes adjacent value distances for each sample. Adds epsilon distance for the last sample.

        Args:
            query_points (_type_): N_rays x N_samples x 3 tensor

        Returns:
            distances N_rays x N_samples x 1 tensor
        """
        distances = torch.linalg.norm(query_points[:, 1:,:] - query_points[:,:-1,:],axis=-1)
        eps_distances_for_final_sample = torch.ones((query_points.shape[0],1),device=self.device)*1e-10
        distances = torch.cat([distances,eps_distances_for_final_sample], dim =-1)
        return distances.unsqueeze(-1)

    def training_step(self,batch,batch_idx):
        ray_origins, ray_directions,rgb_targets = batch

        rgb_values = self.render_rays(ray_origins, ray_directions,self.n_coarse_samples)

        loss = self.mse(rgb_values,rgb_targets)
        psnr = -10.0 * torch.log(loss) / torch.log(10.0)
        self.log("train/loss",loss)
        self.log("train/psnr", psnr)
        return loss 


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=4e-4)


def get_image_rays(camera_pose: torch.Tensor, H:int, W:int, focal_distance:float) -> Tuple[torch.Tensor, torch.Tensor]:
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
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = camera_pose[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


if __name__ == "__main__":
    # nerf = NeRF()
    #nerf.render_rays(np.zeros((3,3)),None,10)

    nerf = NeRFNetwork()
    x = torch.rand(4,6)
    y = torch.rand(4,1)
    optimizer = torch.optim.Adam(nerf.parameters(),lr=3e-4)
    for _ in range(30):
        yhat=nerf(x)[0]
        loss = torch.mean((y-yhat)**2)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    nerf = NeRFNetwork()
    x = torch.rand(4,6)
    y = torch.rand(4,1)
    optimizer = torch.optim.Adam(nerf.parameters(),lr=3e-4)
    for _ in range(30):
        yhat=nerf(x)[0]
        loss = torch.mean((y-yhat)**2)
        print(loss)
        loss.backward()

        print(f"grad={nerf.rgb_head.bias.grad}")
        optimizer.step()
        optimizer.zero_grad()

    