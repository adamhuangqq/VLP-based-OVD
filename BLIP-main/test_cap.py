from PIL import Image
import requests
import torch
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use {device}------------------------------')
def load_demo_image(image_size,device,image_path):
    img_url =  image_path
    raw_image = Image.open(img_url).convert('RGB')   

    w,h = raw_image.size
    # display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

from models.blip import blip_decoder

model_url = 'weights/downstream/model_base_caption_capfilt_large.pth'
image_size = 384

model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)


num = 0
image_path = 'datasets/test_images/00a011632563dae2.jpg'
image = load_demo_image(image_size=image_size,device=device,image_path=image_path)

    
    
with torch.no_grad():
    # beam search
    caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
    # nucleus sampling
    # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
    print('caption: '+caption[0])
