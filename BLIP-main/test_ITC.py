from PIL import Image
import requests
import torch
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
from IPython.display import display
import numpy as np
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('-------------use %s-----------'%(device))
def load_demo_image(image_size,device,image_path):
    img_url =  image_path
    raw_image = Image.open(img_url).convert('RGB')   

    w,h = raw_image.size
    #display(raw_image.resize((w//5,h//5)))
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


from models.blip_itm import blip_itm

image_size = 384
image_path = 'D:/VSCODE/prompt/save/'
annotation_path = 'D:/VSCODE/prompt/RPN_results/proposal_select/'

model_url = 'weights/downstream/model_base_retrieval_coco.pth'
model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device='cpu')
class_file = 'voc_classes.txt' 
with open(class_file,'r') as f:
    class_list = f.readlines()
    class_list =[x.split('\n')[0] for x in class_list]
caption = class_list
print('text: %s' %caption)
# itm_output = model(image,caption,match_head='itm')
# itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
# print('The image and text is matched with a probability of %.4f'%itm_score)
# save_path = 'D:/VSCODE/prompt/save_result/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# for img in os.listdir(annotation_path):
#     print(img)
#     img_list = glob.glob(image_path + img[:-4] + '*.jpg')
    
#     content = ['null'] * len(img_list)
#     for i in img_list:
#         num = int(i.split('-')[-1][:-4])
#         image = load_demo_image(image_size=image_size,device=device,image_path=i)
#         with torch.no_grad():
#             itc_score = model(image,caption,match_head='itc').numpy().astype(np.float32)
#         argmax_itc = itc_score.argmax()
#         content[num] = caption[argmax_itc]
#     with open(save_path + img[:-4] + '.txt', 'w') as f:
#         for line in content:
#             f.write(line)
#             f.write('\n')
#     #print(itc_score)

for img in os.listdir(image_path):
    print(img)
    image = load_demo_image(image_size=image_size,device=device,image_path=image_path+img)
    with torch.no_grad():
        itc_score = model(image,caption,match_head='itc').numpy().astype(np.float32)
    argmax_itc = itc_score.argmax()
    print(caption[argmax_itc])
