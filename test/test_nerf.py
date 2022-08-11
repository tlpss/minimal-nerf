import unittest

import torch

from minimal_nerf.nerf import NeRFNetwork


class TestNetwork(unittest.TestCase):
    def test_creation_succes(self):
        NeRFNetwork()

    def test_creation_custom_args_success(self):
        NeRFNetwork(4, 128, True, undefined_arg=None)

    def test_forward_batch(self):
        nerf = NeRFNetwork()
        x = torch.rand(20, 6)
        rgb, density = nerf(x)
        self.assertEqual(rgb.shape, (20, 3))
        self.assertEqual(density.shape, (20, 1))


class TestNeRF(unittest.TestCase):
    pass
