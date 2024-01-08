from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.blip_vqa import blip_vqa
def load_demo_image(image_size,device,img_path):
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    raw_image = Image.open(img_path).convert('RGB')   

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image



image_size = 480
img_path = 'datasets/test_images/0bf842fefbfb2770.jpg'
print(img_path)
image = load_demo_image(image_size=image_size, device=device,img_path=img_path)     

model_weights = 'weights/downstream/model_base_vqa_capfilt_large.pth'
    
model = blip_vqa(pretrained=model_weights, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

question = 'Where are the black shorts?'

with torch.no_grad():
    answer = model(image, question, train=False, inference='generate') 
    print('question: '+question)
    print('answer: '+answer[0])
