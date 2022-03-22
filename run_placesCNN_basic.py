# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image

import argparse
import os
from os import walk
import sys
from pathlib import Path
import glob

import pandas as pd
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
ROOT = str(ROOT)

def get_files_in_dir(*extensions, path = './data/train/Japan/labels', show_folders = False, **kwargs):
    'Function to return all files in a directory with option of filtering for specific filetype extension'
    result = []
    for extension in list(extensions):
        files = next(walk(path), (None, None, []))[2]
        if 'fullpath' in kwargs and kwargs['fullpath']:
            files = [path + '/' + f for f in files if f.endswith(extension)]
            result.extend(files)
        else: 
            files = [f for f in files if f.endswith(extension)]
            result.extend(files)
    if show_folders:
        folders = next(walk(path), (None, None, []))[1]
        folders = [folder for folder in folders if folder[0] != '.']
        return result, folders
    return result

def parse_opt():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--folder', type=str, default='images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--model_type', type=str, default='resnet18', help='available model types=[alexnet,resnet18,resnet50,densenet161]')    
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    print('opt: ', opt)
#     print_args(FILE.stem, opt)
    return opt

def classify_scene_for_image(model, classes, img_name = ''):
    
    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the test image
    if not os.access(img_name, os.W_OK):
        img_url = 'http://places.csail.mit.edu/demo/' + img_name
        os.system('wget ' + img_url)
    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))
    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
#     print('Prediction on {}'.format(img_name))
    # output the prediction
    preds = {}
    for i in range(0, 5):
#         print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
        preds[f'pred{i}'] = classes[idx[i]]
        preds[f'prob{i}'] = float(probs[i].data)
    return preds
        
def main(options): #, folder = ROOT  / 'images'
    '''
    Function loading model weights to from Places365 MIT CSAIL lab webpage http://places2.csail.mit.edu/models_places365/
    Inputs:
    architecture = choose one of ['alexnet','resnet18','resnet50','densenet161']
    '''
    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % options.model_type #arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[options.model_type](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    print(options.folder, type(options.folder))
    image_files = get_files_in_dir('jpg', 'png', path = options.folder, show_folders = False, fullpath = True)
    df_data = []
    for image_file in image_files:
        preds = classify_scene_for_image(model = model, classes = classes, img_name = image_file) #**vars(options)
        df_data.append(preds)
    df = pd.DataFrame(data = df_data)
    df.to_csv('results/scene_classification_results.csv', sep = ',')
    print(df)

if __name__ == '__main__':
    options = parse_opt()
    main(options = options)
