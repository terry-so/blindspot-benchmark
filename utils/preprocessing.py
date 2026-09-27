from PIL import Image
from torchvision import transforms

def lpips_image_preprocess(PIL_image):
    image = PIL_image.convert('RGB')
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    normalize_transform = transforms.Normalize(0.5,0.5)
    normalized_image = normalize_transform(tensor_image)
    return normalized_image.unsqueeze(0).to('cuda')

def lpips_distance(first_image_path, second_image_path):
    """Return the LPIPS distance between two images (smaller means more similar)."""
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchvision.transforms import functional as F

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with Image.open(first_image_path) as image:
        first = lpips_image_preprocess(image)
    with Image.open(second_image_path) as image:
        second = lpips_image_preprocess(image)

    second = F.resize(second, first.shape[-2:])
    metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
    metric.eval()
    with torch.no_grad():
        return metric(first, second).item()
    
