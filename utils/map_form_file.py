import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
import shutil



if __name__=='__main__':
    class_path = 'model_data/car_classes.txt'
    class_names,_ = get_classes(class_path)
    path = 'results/car/'
    #get_coco_map(class_names, path)
    for score in [0.5]:
        temp_map = get_map(0.5, False, score, path = path)
        shutil.rmtree(path+'results/')
    #print(temp_map)

