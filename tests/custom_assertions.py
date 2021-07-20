import numpy as np
import torch


class CustomAssertions:
    def assertAreTensors(self, *args):
        if not all([torch.is_tensor(arg) for arg in args]):
            raise AssertionError("All values should be of type torch.Tensor")

    def assertTensorsAlmostEqual(self, expected, actual, decimal=5):
        """
        Test tensors are almost equal (EPS = 1e-5 by default)
        """
        np.testing.assert_almost_equal(expected, actual, decimal=decimal)
