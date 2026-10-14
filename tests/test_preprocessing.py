import unittest
from PIL import Image
try:
    import torch
    import torchvision
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'PyTorch/torchvision not installed')
class PreprocessingTests(unittest.TestCase):
    def test_endpoints_shape_and_dtype(self):
        from utils.preprocessing import lpips_image_preprocess
        for color,value in [('black',-1),('white',1)]:
            tensor=lpips_image_preprocess(Image.new('RGB',(32,32),color),device='cpu')
            self.assertEqual(tuple(tensor.shape),(1,3,32,32))
            self.assertEqual(tensor.dtype,torch.float32)
            self.assertTrue(torch.all(tensor==value))
    def test_invalid_values_are_rejected(self):
        from utils.preprocessing import validate_lpips_tensor
        for value in [float('nan'),float('inf'),1.01,-1.01]:
            with self.assertRaises(ValueError):
                validate_lpips_tensor(torch.full((1,3,32,32),value))
        with self.assertRaises(ValueError):validate_lpips_tensor(torch.zeros((3,32,32)))
        with self.assertRaises(ValueError):validate_lpips_tensor(torch.zeros((1,3,32,32),dtype=torch.float64))
    def test_roundoff_is_corrected(self):
        from utils.preprocessing import validate_lpips_tensor
        value=torch.full((1,3,32,32),1+torch.finfo(torch.float32).eps)
        self.assertTrue(torch.all(validate_lpips_tensor(value)==1))
