from transformers import BlipProcessor, BlipForConditionalGeneration
from tkinter import filedialog
from PIL import Image
import torch
import tqdm
import glob
import time
import os

print("\n\nFed the Stuff... Jus' Hit Enter if there's None")
preCon = input("PreCaption Contents Yo: ") + ' '
postCon = input("PostCaption Contents Yo: ")

t1 = time.time()
print("Select the imgDirectory Yo!")
imgDir = filedialog.askdirectory() + '//'
print("---->", imgDir)

print("Select the Out captionDirectory Yo!")
capDir = filedialog.askdirectory() + '//'
print("---->", capDir)

x = [glob.glob(imgDir+y) for y in ['*.jpg', '*.png', '*.tiff', '*.bmp', '*.jpeg']]
x = sum(x , [])

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

for i in tqdm.tqdm(x, desc = "Captionin' the Images Yo!", colour = 'red'):
    rawImg = Image.open(i).convert('RGB')
    inputs = processor(rawImg, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    genOut = processor.decode(out[0], skip_special_tokens=True).replace('there is a', 'a')
    txtOut = preCon + genOut + postCon
    print(txtOut)
    i = i.replace(imgDir, capDir)
    i = ''.join([str(x) for x in i.split('.')[:-1]])
    f = open(i + '.txt', 'w')
    f.write(txtOut)
    f.close()

t2 = time.time()
print("\nCompleteExecTime: ", (t2-t1))