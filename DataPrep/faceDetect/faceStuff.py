from colorama import Fore, Style
import matplotlib.pyplot as plt
from tkinter import filedialog
from ultralytics import YOLO
import pandas as pd
import numpy as np
import random
import torch
import tqdm
import glob
import cv2
import os

print(Fore.YELLOW+Style.BRIGHT+"\n\nSelect in-imgDir Yo !"+Fore.RESET)
#imgDir = filedialog.askdirectory()
imgDir = 'P:\\OpenSourceContribution\\VirtualME\\DataPrep\\prepData\\oldDim'
print("---->", imgDir)
if "\\" in imgDir:
    imgDir = imgDir + '\\'
else:
    imgDir = imgDir + '/'

print(Fore.BLUE+Style.BRIGHT+"\n\nSelect the Trained Weight File Yo!"+Fore.RESET)
#inWeight = filedialog.askopenfilename(filetypes=[("pt Weight File", "*.pt")])
inWeight = 'P:\\OpenSourceContribution\\VirtualME\\DataPrep\\faceDetect\\trainedWeights\\faceDetectLast.pt'

numStuff = {'vFace': 0}
clsStuff = {0: 'vFace'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelStuff = YOLO(inWeight).to(device)

imgStuff = [glob.glob(imgDir+y) for y in ['*.jpg', '*.png', '*.tiff', '*.bmp', '*.jpeg']]
imgStuff = sum(imgStuff , [])

print(imgStuff)

outVal = []
for inImg in tqdm.tqdm(imgStuff, desc = "Flowin' through images Yo!", colour = "red"):
    outStuff = modelStuff(inImg, save_conf=True)
    outStuff = outStuff[0].boxes.to("cpu")
    print(outStuff)
    print(len(outStuff.cls))
    if len(outStuff.cls)==0:
        print("Daaawwm !")
    else:
        imgWidth = int(outStuff.orig_shape[0])
        imgHeight = int(outStuff.orig_shape[1])
        xCen = float(outStuff.xywhn[0][0])
        yCen = float(outStuff.xywhn[0][1])
        clsWidth = float(outStuff.xywhn[0][2])
        clsHeight = float(outStuff.xywhn[0][3])
        xUnits = int(clsWidth * imgWidth)
        yUnits = int(clsHeight * imgHeight)
        xMin = int(((xCen * imgWidth * 2) - xUnits) / 2)
        yMin = int(((yCen * imgHeight * 2) - yUnits) / 2)
        xMax = xMin + xUnits
        yMax = yMin + yUnits
        imgAr = cv2.imread(inImg, 0)
        print(imgAr)
        print(imgAr.shape)
        print("xMin {} xMax {} yMin {} yMax {}".format(xMin, xMax, yMin, yMax))
        faceStuff = imgAr[xMin:xMax, yMin:yMax]
        print(faceStuff)
        print(faceStuff.shape)
        plt.imshow(faceStuff)
        plt.show()