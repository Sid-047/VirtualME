import os
import tqdm
import shutil
from tkinter import filedialog
from colorama import Fore, Style

print(Fore.BLUE+Style.BRIGHT+"\nSelect the outDir Yo!"+Fore.RESET)
outDir = filedialog.askdirectory()

if "\\" in outDir:
    outDir = outDir + '\\'
    imgDir = outDir + 'imgSet' +'\\'
else:
    outDir = outDir + '/'
    imgDir = outDir + 'imgSet' +'/'

if not os.path.isdir(imgDir):
   os.makedirs(imgDir)

for i,j,k in os.walk(os.getcwd(), topdown=True):
    print(i)
    for f in tqdm.tqdm(k, colour = 'red'):
        if '.png' in f or '.jpg' in f:
            if "\\" in i:
                inPath = i+'\\'+f
            else:
                inPath = i+'/'+f
            outPath = outDir + f
            shutil.copy(inPath, outPath)