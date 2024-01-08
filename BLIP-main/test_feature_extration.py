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


from models.blip import blip_feature_extractor
image_path = 'merlion.png'
image_size = 224
image = load_demo_image(image_size=image_size, device=device, image_path=image_path)     

model_url = 'weights/downstream/model_base_retrieval_coco.pth'
    
model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

caption = 'i want to go to school'

multimodal_feature = model(image, caption, mode='multimodal')[0,0]
image_feature = model(image, caption, mode='image')[0,0]
text_feature = model(image, caption, mode='text')[0,0]
print(model(image, caption, mode='multimodal').shape)
