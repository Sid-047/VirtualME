from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tkinter import filedialog
import random
import torch
import tqdm
import glob
import time
import os

t1 = time.time()
print("Select the Directory Yo!")
imgDir = filedialog.askdirectory() + '\\'
print("---->", imgDir)
x = [glob.glob(imgDir+y) for y in ['*.jpg', '*.png', '*.tiff', '*.bmp', '*.jpeg']]
x = sum(x , [])
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

for i in tqdm.tqdm(x, desc = "captionin' the Images Yo!", colour = 'red'):
    rawImg = Image.open(i).convert('RGB')
    inputs = processor(rawImg, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True).replace('here is a', 'a'))

t2 = time.time()
print("\nCompleteExecTime: ", (t2-t1))