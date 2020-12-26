# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 01:00:23 2020

@author: Stark_3000
"""
import streamlit as st
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import datasets,transforms
from glob import glob
import os
from PIL import Image
from matplotlib import patches
from torch.utils.data import Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader


st.title("Wanna know the future")
upload = st.file_uploader("Put your image here",type="jpg")

image = Image.open(upload)

st.image(image,caption="Image uploaded",use_column_width=True)



weights = 'F://Datasets//Wheat Detection//fasterrcnn_resnet50_fpn (2).pth'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False,pretrained_backbone=False)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
num_class = 2 #wheats and background

in_feature = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_feature,num_class) #changin the pretrained head with a new one

model.load_state_dict(torch.load(weights))
model.eval()


x = model.to(device)


def collate_fn(batch):
    return tuple(zip(*batch))


test_dataset = Wheatdatasets(sample,test_dir,get_transform())



test_dataloader = DataLoader(
test_dataset,
batch_size=4,
shuffle =False,
num_workers =4,drop_last = False,
collate_fn = collate_fn)



def format_string(boxes,score):
    pred_strings = []
    for j in zip(score,boxes):
        
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
        
    return " ".join(pred_strings)

detection_threshold = 0.5
results = []

for images, image_ids in test_dataloader:

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        result = {
            'image_id': image_id,
            'PredictionString': format_string(boxes, scores)
        }

        
        results.append(result)
        
sample = images[0].permute(1,2,0).cpu().numpy()
boxes = outputs[0]['boxes'].data.cpu().numpy()
score = outputs[0]['scores'].data.cpu().numpy()

boxes = boxes[score >= detection_threshold].astype(np.int32)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 2)
    
ax.set_axis_off()
ax.imshow(sample)