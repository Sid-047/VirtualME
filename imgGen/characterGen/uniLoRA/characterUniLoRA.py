from diffusers import DiffusionPipeline
from colorama import Fore, Style
from tkinter import filedialog
import random
import torch
import glob
import tqdm
import time
import os

charList = []
print(Fore.RED+Style.BRIGHT+"\n\nEnter the Charaters Yo n Jus' Enter '7' to End!"+Fore.RESET)
while True:
    x = input()
    if x!='7':
        charList.append(x.lower())
    else:
        break

print(Fore.BLUE+Style.BRIGHT+"\n\nSelect the fineTuned SafeTensor File Yo!"+Fore.RESET)
inLoRA = filedialog.askopenfilename(filetypes=[("LoRA SafeTensor", "*.safetensors")])
if "\\" in inLoRA:
    fileLoRA = inLoRA.split('\\')[-1]
    inDir = inLoRA.replace(fileLoRA , '')
else:
    fileLoRA = inLoRA.split('/')[-1]
    inDir = inLoRA.replace(fileLoRA , '')

w = 768
h = 1024
gScale = 7.5
valSeed = 42
nPrompt = "wrinkles, forehead lines, worst quality, Disfigured, logo, Malformed, kitsch, extra legs,\
        malformed limbs, Replicate, long fingers, Distorted, watermark, jpeg artifacts, \
        Extra arms, Tiling, worst face, disgusting, mangled, Outside the picture, extra fingers, \
        Deformed, Mistake, Blank background, bad anatomy, extra limbs, distorted face, \
        worst thigh, Storyboard, Low quality, disfigured, Branding, Written language, \
        blur, Logo, Signature, Mutated hands, extra eyes, Unsightly, Unreal engine, \
            Missing fingers, username, three thigh, blurry, low quality, Extra hands, \
            Watermark, Draft, overexposed, dehydrated, ugly eyes, Shortened, \
            too many fingers, Poorly drawn face, Kitsch, floating limbs, \
            Identifying mark, Mutilated, 2 heads, Mutation, Out of frame, \
            Script, low contrast, Poorly drawn feet, irregular face, \
                bad arms, Split image, grainy, poorly drawn hands, \
                unnatural pose, monochrome, Extra fingers, tiling, \
                Printed words, Squint, sketch ,duplicate, signature, \
                Indistinct, Beyond the borders, text, Blurry, \
                    Ugly, fused fingers, long neck, bad composition, \
                    missing limb, draft, Improper scale, Amputee, \
                    mutilated, extra digits, mutated hands, Reproduce,\
                    childish, huge eyes, poorly framed, Bad illustration,\
                    mutation, Duplicated features, underexposed,\
                        three hands, Macabre, Text, body out of frame, \
                        long body, malformed hands, oversaturated, \
                        low-res, Bad anatomy, cropped, three feet, amateur, \
                        Revolting dimensions, Long neck, deformed, out of frame, \
                            deformed face, surreal, fused feet, ugly fingers, Render, \
                            low res, bad proportions, Bad proportions, Cropped, \
                            Unfocused, Mark, beginner, gross proportions, Trimmed, \
                            Sign, Off-screen, Hazy, Morbid, error, Body out of frame, \
                            cut off, extra thigh, duplicate, Incorrect ratio, extra limb, \
                                horn, low effort, Unattractive, deformed fingers, Missing hands, \
                                grain, fused crus, banner, Poorly drawn hands, blurred faces, \
                                missing legs, fused face, Autograph, Missing arms, extra arms, \
                                Fault, ugly, poorly drawn feet, irrregular body shape, squint, old, \
                                    Missing legs, mutated, Disproportioned, Repellent, out of focus, \
                                    Misshapen, lowres, bad art, Flaw, poorly drawn, Duplicate, Gross proportions, \
                                    Pixelated, amputation, blurred, disconnected limbs, fused thigh, horror, \
                                    bad hands, three crus, cloned face, morbid, Extra limbs, worst feet, \
                                    geometry, Cut off, three legs, Extra legs, Boring background, poor, \
                                        missing fingers, missing arms, Grainy, extra crus, normal quality, \
                                        Dismembered, poorly drawn face, Grains, 2 faces, Incorrect physiology, \
                                        Low resolution, Fused fingers, bad face"

gSeed = torch.Generator(device="cuda").manual_seed(valSeed)

if not os.path.isdir("genDir"):
    os.makedirs("genDir")
os.chdir("genDir")

if not os.path.isdir("runTwo"):
    os.makedirs("runTwo")
os.chdir("runTwo")

t1 = time.time()

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = pipe.to("cuda")
nSampling = 9
pipe.load_lora_weights(inLoRA)
pipe.fuse_lora()
imgDir = inLoRA.replace(inDir, "").split('.')[0]
for j in tqdm.tqdm(list(range(68)), desc = "Samplin' iters Yo!", colour = "red"):
    nSampling+=1
    random.shuffle(charList)
    charStuff = random.choice(charList)
    pPrompt = "vSid a hyper-realistic man as a {}, best quality, upper body,\
        center of frame, high resolution, looking at viewer".format(charStuff)
    img = pipe(prompt = pPrompt, negative_prompt = nPrompt, height = h, width = w, 
            num_inference_steps = nSampling, generator = gSeed, guidance_scale = gScale).images[0]
    img.save("{}.png".format(str(nSampling)+'_'+imgDir+'_'+str(time.time())))

t2 = time.time()
print("\nExecTime: ", (t2-t1))