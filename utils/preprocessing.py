from PIL import Image
from torchvision import transforms

def lpips_image_preprocess(PIL_image):
    image = PIL_image.convert('RGB')
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    normalize_transform = transforms.Normalize(0.5,0.5)
    normalized_image = normalize_transform(tensor_image)
    return normalized_image.unsqueeze(0).to('cuda')
    
