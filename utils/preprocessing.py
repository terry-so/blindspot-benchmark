from PIL import Image


def validate_lpips_tensor(tensor):
    import torch

    if (
        tensor.ndim != 4
        or tensor.shape[0] != 1
        or tensor.shape[1] != 3
        or min(tensor.shape[-2:]) < 1
    ):
        raise ValueError("LPIPS expects a nonempty 1x3xHxW tensor")
    if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise ValueError("LPIPS expects finite float32 values")
    tolerance = 8 * torch.finfo(torch.float32).eps
    if tensor.min() < -1 - tolerance or tensor.max() > 1 + tolerance:
        raise ValueError("LPIPS tensor is outside [-1,1]")
    # Only floating-point roundoff is corrected, after checking the range.
    return tensor.clamp(-1, 1)


def lpips_image_preprocess(PIL_image, device=None):
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tensor = pil_to_tensor(PIL_image.convert("RGB")).to(dtype=torch.float32)
    tensor = tensor.div(255.0).mul(2.0).sub(1.0).unsqueeze(0)
    return validate_lpips_tensor(tensor).to(device)


def lpips_distance(first_image_path, second_image_path):
    from evaluators.auto_grader import image_distance

    with Image.open(first_image_path) as first, Image.open(second_image_path) as second:
        return image_distance(first, second)
