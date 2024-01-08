import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from frcnn import FRCNN

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__=='__main__':
    class_path = 'model_data/voc_classes.txt'
    class_names,_ = get_classes(class_path)
    path = 'voc_map_out/'
    #get_coco_map(class_names, path)
    temp_map = get_map(0.5, False, path = path)