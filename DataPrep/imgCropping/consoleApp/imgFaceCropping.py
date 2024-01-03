from tkinter import filedialog
from deepface import DeepFace
import random
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



t2 = time.time()
print("\nCompleteExecTime: ", (t2-t1))