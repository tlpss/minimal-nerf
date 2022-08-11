import argparse
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.random
import wandb
from pytorch_lightning.loggers import WandbLogger


class NeRFNetwork(nn.Module):
    """
    NeRF network as described in the original NeRF paper.
    density depends only on position; the emitted radiance on both position and view angle.

    The viewing distance is represented by a unit vector so the network input is 6D (x,y,z,dx, dy, dz)
    and the output is a tuple (RGB,density)
    """

    @staticmethod
    def add_model_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("NerfNetwork")
        parser.add_argument(
            "--depth", help="number of layers from position to the feature vector and density layer", type=int
        )
        parser.add_argument(
            "--width", help="width of the layers from the position to the feature vector and density layer", type=int
        )
        parser.add_argument(
            "--use_skip_connection",
            help="Use a skip connection halfway through the layers between position and density/feature vector",
            action=argparse.BooleanOptionalAction,
        )
        return parent_parser

    def __init__(self, depth=4, width=128, use_skip_connection: bool = True, **kwargs):

        super().__init__()
        assert depth % 2 == 0, "depth should be even"

        self.depth = depth
        self.width = width
        if use_skip_connection:
            self.skips = [depth // 2]  # lazy
        else:
            self.skips = []

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.feature_layers = nn.ModuleList([nn.Linear(3, self.width)])
        for i in range(self.depth - 1):
            depth = i + 1
            if depth in self.skips:
                self.feature_layers.append(nn.Linear(self.width + 3, self.width))
            else:
                self.feature_layers.append(nn.Linear(self.width, self.width))

        self.density_head = nn.Linear(self.width, 1)
        self.feature_head = nn.Linear(self.width, self.width)
        self.rgb_layer = nn.Linear(self.width + 3, self.width // 2)
        self.rgb_head = nn.Linear(self.width // 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: N_rays x N_samples x 6 (or at least C... x 6)

        Returns:
            rgb: N_rays x N_samples x 3
            density: N_rays x N_samples x 1
        """

        position, view_direction = torch.split(x, 3, dim=-1)
        x = position
        for i, layer in enumerate(self.feature_layers):
            if i in self.skips:
                x = torch.cat([x, position], dim=-1)
            x = layer(x)
            x = self.relu(x)

        density = self.density_head(x)
        density = self.relu(density)  # constrain to [0, Inf.]

        features = self.feature_head(x)
        features = self.relu(features)

        x = torch.cat([features, view_direction], dim=-1)
        x = self.rgb_layer(x)
        x = self.relu(x)
        rgb = self.rgb_head(x)
        rgb = self.sigmoid(rgb)  # constrain to [0,1]

        return rgb, density


class NeRF(pl.LightningModule):
    """A minimal implementation of Vanilla NeRFs as published in "Representing scenes as Neural Radiance Fields (NeRF) for novel view synthesis" (https://arxiv.org/abs/2003.08934)
    This system learns to represent such a single scene from a set of images and corresponing camera extrinsics and intrinsics.
    """

    # TODO: introduce hierarchical sampling

    @staticmethod
    def add_model_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("NeRF")
        parser.add_argument(
            "--n_coarse_samples",
            help="number of coarse samples to take for each ray that needs to be rendered. More will result in more accurate renders but results in slower rendering.",
            type=int,
        )
        parser.add_argument(
            "--min_render_distance",
            help="minimal rendering distance for each ray. important to configre this correctly",
            type=float,
        )
        parser.add_argument(
            "--max_render_distance",
            help="maximal rendering distance for each ray. Important to configure this correctly.",
            type=float,
        )
        parser.add_argument(
            "--debug",
            help="log debug information such as model gradients and rendering steps ",
            action=argparse.BooleanOptionalAction,
        )

        parser = NeRFNetwork.add_model_argparse_args(parser)

        return parent_parser

    def __init__(
        self,
        min_render_distance: float = 2.0,
        max_render_distance: float = 6.0,
        n_coarse_samples: int = 64,
        debug: bool = False,
        **kwargs,
    ) -> None:
        self.debug = debug

        super().__init__()

        # self.save_hyperparameters()

        self.mse = nn.MSELoss()

        # TODO: get these from colmap?
        # these values are for the Blender scenes.
        self.min_render_distance = min_render_distance
        self.max_render_distance = max_render_distance

        self.n_coarse_samples = n_coarse_samples
        self.nerf = NeRFNetwork(**kwargs)
        if self.debug and isinstance(self.logger, WandbLogger):
            # log gradients
            self.logger.watch(self.nerf, log_freq=10)

        # self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nerf(x)

    def render_view(self, camera_pose: torch.Tensor, height: int, width: int, focal_distance: float) -> torch.Tensor:
        """Renders a view using a pinhole camera model with the specified camera pose, image dimensions and focal distance.

        Args:
            camera_pose (torch.Tensor): 4x4 camera pose - extrinsics matrix
            height (int): height of the view to be rendered
            width (int): width of the view to be rendered
            focal_distance (float): focal distance of the view to be rendered, expressed in pixels.

        Returns:
            (torch.Tensor): HxWx3 image
        """

        rays_origin, rays_direction = get_image_rays(
            camera_pose.cpu(), height, width, focal_distance, device=self.device
        )
        rendered_rgbs = self.render_rays(
            rays_origin.flatten(end_dim=-2), rays_direction.flatten(end_dim=-2), self.n_coarse_samples
        )
        rendered_view = rendered_rgbs.reshape((height, width, 3))
        return rendered_view

    def render_rays(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, n_coarse_samples: int
    ) -> torch.Tensor:
        """Renders the RGB value for the specified rays by taking n_coarse_samples along each ray,
         querying the network for the rgb and density at each sample along the rays and combining these values in a differentiable way,
          which allows to backpropagat the eventual difference between the rendered rgb value and true (pixel) value all the way to the implicit representation network (the NeRF).


        The sample points are determined by uniformly sampling in bins along the ray, between the specified minimal render distance and maximal render distance.
        These parameters are very important, as they effectively determine what part of the 3D space is queried and hence backpropagated to.

        Withouth the uniform sampling in the bins, the network would only learn a grid and would struggle to interpolate to new views.

        Args:
            ray_origins (torch.Tensor): N_rays x 3
            ray_directions (torch.Tensor): N_rays x 3, normalized direction vectors
            n_coarse_samples (int): number of coarse samples

        Returns:
            torch.Tensor: N_rays,3, the rendered RGB values for each ray.
        """

        network_query_points, z_sample_points = self._compute_query_points(
            ray_origins, ray_directions, n_coarse_samples
        )
        network_query_inputs = torch.cat(
            [network_query_points, ray_directions.unsqueeze(1).expand(-1, n_coarse_samples, -1)], dim=-1
        )
        queried_rgbs, queried_densities = self.forward(network_query_inputs)

        rendered_rgbs = self._volume_render_rays(z_sample_points, queried_rgbs, queried_densities)

        return rendered_rgbs

    def _compute_query_points(
        self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, n_coarse_samples: int
    ) -> torch.Tensor:
        """

        Args:
            ray_origins (_type_): N_rays x 3
            ray_directions (_type_): N_rays x 3
            n_coarse_samples (_type_):

        Returns:
            query point: n_rays x n_coarse_samples x 3
            z_distances: n_rays x n_coarse_samples x 1
        """
        bins = torch.linspace(
            self.min_render_distance, self.max_render_distance, steps=n_coarse_samples + 1, device=self.device
        )
        bin_width = (self.max_render_distance - self.min_render_distance) / n_coarse_samples
        uniform_bin_samples = torch.rand(ray_origins.shape[0], n_coarse_samples, device=self.device)
        uniform_bin_samples *= bin_width
        z_sample_points = (
            bins[:-1].unsqueeze(0).unsqueeze(-1).expand(ray_directions.shape[0], -1, -1)
        )  # + uniform_bin_samples

        expanded_ray_directions = ray_directions.unsqueeze(1).expand(
            -1, n_coarse_samples, -1
        )  # (N_rays x N_samples x 3)
        query_points = ray_origins.unsqueeze(1) + expanded_ray_directions * z_sample_points
        return query_points, z_sample_points

    def _volume_render_rays(self, z_sample_points, queried_rgbs, queried_densities):
        """Differentiable Volume Rendering using the queried rgb and density values.

         rgb@ ray = sum(transmittance x infinitesimal density x rgb, for each sample along the ray)
         where transmittance = the probability that the ray was able to travel this far w/o hitting any "volume"


        Args:
            z_sample_points (_type_): N_rays x N_samples x 1
            queried_rgbs (_type_): N_rays x N_samples x 3
            queried_densities (_type_): N_rays x N_samples x 1

        Returns:
            _type_: _description_
        """
        queried_densities = (
            queried_densities  # TODO: check if random noise helps? +  torch.randn_like(queried_densities) * 0.01
        )
        queried_densities = self.nerf.relu(queried_densities)

        distances = self._compute_adjacent_distances(z_sample_points)

        neg_weighted_densities = -distances * queried_densities + 1e-10  # eps for numerical stability in exp later on

        zeros_shape = (neg_weighted_densities.shape[0], 1, 1)
        transmittances = torch.cat([torch.zeros(zeros_shape, device=self.device), neg_weighted_densities], dim=-2)[
            :, :-1, :
        ]
        accumulated_transmittances = torch.exp(torch.cumsum(transmittances, dim=-2))

        alpha = 1 - torch.exp(neg_weighted_densities)
        weights = alpha * accumulated_transmittances

        rgbs = torch.sum(weights * queried_rgbs, dim=-2)

        if self.logger and self.debug:
            wandb.log(
                {
                    "debug/queried_densities": wandb.Histogram(queried_densities.detach().cpu().numpy()),
                    "debug/queried_rgbs": wandb.Histogram(queried_rgbs.detach().cpu().numpy()),
                    "debug/alpha": wandb.Histogram(alpha.detach().cpu().numpy()),
                    "debug/accum_transmittances": wandb.Histogram(accumulated_transmittances.detach().cpu().numpy()),
                    "debug/distances": wandb.Histogram(distances.detach().cpu().numpy()),
                    "debug/rendered_rgbs": wandb.Histogram(rgbs.detach().cpu().numpy()),
                }
            )
        return rgbs

    def _compute_adjacent_distances(self, z_value_samples: torch.Tensor) -> torch.Tensor:
        """computes the distance to the next z_value sample for a batch of sampled rays.
        Adds epsilon distance for the last sample (for which there is no distance to the next sample).

        Args:
            z_value_samples (torch.Tensor): N_rays x N_samples x 1 tensor

        Returns:
            distances N_rays x N_samples x 1 tensor
        """
        z_value_samples = z_value_samples.squeeze(-1)
        distances = z_value_samples[:, 1:] - z_value_samples[:, :-1]
        # effictively ignores the last sample as the distance is used as negative exponent
        eps_distances_for_final_sample = torch.ones((z_value_samples.shape[0], 1), device=self.device) * 1e10
        distances = torch.cat([distances, eps_distances_for_final_sample], dim=-1)
        return distances.unsqueeze(-1)

    def training_step(self, batch, batch_idx):
        """
        Training step: takes a batch of shuffled rays and corresponding RGB values,
        then uses the NeRF to perform differentiable volume rendering and computes the MSE between the rendered RGB values and the ground truth values.


        Args:
            batch (_type_): (ray_origins, ray_directions, rgb_targets): each N_rays x 3 tensors
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        ray_origins, ray_directions, rgb_targets = batch

        rgb_values = self.render_rays(ray_origins, ray_directions, self.n_coarse_samples)

        loss = self.mse(rgb_values, rgb_targets)

        psnr = -10.0 * torch.log(loss) / torch.log(torch.tensor([10.0], device=self.device))
        self.log("train/loss", loss)
        self.log("train/psnr", psnr)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        """Validation step in which an entire image from the train dataset is rendered and compared to the original.

        Args:
            batch (torch.Tensor): a tuple (image,pose,focal_length)
            batch_idx (_type_): _description_
        """
        image, pose, focal_length = batch
        image = image[0]
        pose = pose[0]
        focal_length = focal_length.item()
        with torch.no_grad():
            rendered_image = self.render_view(pose, image.shape[0], image.shape[1], focal_length)
            rendered_image = rendered_image.cpu().numpy()
            image = image.cpu().numpy()

        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key=f"train_{batch_idx}", images=[image, rendered_image], caption=["train", "rendered"]
            )

        # TODO: log PSNR/ SSIM / MSE

    def configure_optimizers(self):
        # TODO: introduce scheduler as in paper
        return torch.optim.Adam(self.parameters(), lr=5e-4)


### util functions
def get_image_rays(
    camera_pose: torch.Tensor, H: int, W: int, focal_distance: float, device="cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the rays for each pixel in an image, assuming a perfect pinhole camera (so the intrinsics reduce to the focal distance)

    function based on https://github.com/yenchenlin/nerf-pytorch
    Args:
        camera_pose (torch.Tensor): 4x4 homogeneous matrix describing camera pose in world
        height (int): _description_
        width (int): _description_
        focal_distance (float): focal distance measured in pixels.

    Returns:
        ray_origins: (torch.Tensor): HxWx3  origin positions
        ray_direction (torch.Tensor): HxWx3 normalized direction vectors
    """

    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - W // 2) / focal_distance, -(j - H // 2) / focal_distance, -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * camera_pose[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # make sure all rays are normalized!
    direction_norm = torch.linalg.norm(rays_d, dim=-1)
    rays_d = rays_d / direction_norm.unsqueeze(-1)

    # Translate camera frame's origin to the world frame.
    rays_o = camera_pose[:3, -1].expand(rays_d.shape)

    return rays_o.to(device), rays_d.to(device)
