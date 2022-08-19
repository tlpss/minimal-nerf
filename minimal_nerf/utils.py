import math

import torch


def get_z_translation_matrix(translation):
    matrix = torch.eye(4)
    matrix[2, 3] = translation
    return matrix


def get_phi_rotation_matrix(angle):
    matrix = torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, math.cos(angle), -math.sin(angle), 0],
            [0, math.sin(angle), math.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )
    return matrix


def get_theta_rotation_matrix(angle):
    matrix = torch.Tensor(
        [
            [math.cos(angle), 0, -math.sin(angle), 0],
            [0, 1, 0, 0],
            [math.sin(angle), 0, math.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )
    return matrix


def get_camera_frame_convention_matrix():
    return torch.Tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


# todo: figure out how this works?
def get_spherical_poses(radius: float, phi_range: float, steps: int) -> torch.Tensor:
    poses = [
        get_camera_frame_convention_matrix()
        @ get_theta_rotation_matrix(i * phi_range / steps)
        @ get_phi_rotation_matrix(0.0)
        @ get_z_translation_matrix(radius)
        for i in range(steps)
    ]

    return poses


if __name__ == "__main__":
    print(get_spherical_poses(2.0, math.pi, 3))
