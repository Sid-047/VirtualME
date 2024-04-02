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
imgDir = filedialog.askdirectory()
print("---->", imgDir)
if "\\" in imgDir:
    imgDir = imgDir + '\\'
else:
    imgDir = imgDir + '/'

print(Fore.BLUE+Style.BRIGHT+"\n\nSelect the Trained Weight File Yo!"+Fore.RESET)
inWeight = filedialog.askopenfilename(filetypes=[("pt Weight File", "*.pt")])

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
        xMin = int(outStuff.xyxy[0][0])
        yMin = int(outStuff.xyxy[0][1])
        xMax = int(outStuff.xyxy[0][2])
        yMax = int(outStuff.xyxy[0][3])
        imgAr = cv2.imread(inImg, 0)
        print(imgAr)
        print(imgAr.shape)
        print("xMin {} xMax {} yMin {} yMax {}".format(xMin, xMax, yMin, yMax))
        faceStuff = imgAr[yMin:yMax, xMin:xMax]
        print(faceStuff)
        print(faceStuff.shape)
        plt.imshow(faceStuff, cmap='gray')
        plt.show()