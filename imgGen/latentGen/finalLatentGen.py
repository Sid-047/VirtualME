from diffusers import DiffusionPipeline
from colorama import Fore, Style
from tkinter import filedialog
from PIL import Image
from tqdm import tqdm
import torch
import time
import os

print(Fore.BLUE+Style.BRIGHT+"\n\nSelect the fineTuned SafeTensor File Yo!"+Fore.RESET)
inLoRA = filedialog.askopenfilename(filetypes=[("LoRA SafeTensor", "*.safetensors")])

w = 768
h = 768 
gScale = 7.5
valSeed = 42
nSampling = 61
pPrompt = "vSid a hyper-realistic man statue, best quality, upper body, \
    looking at viewer, simple background"
nPrompt = "worst quality, Disfigured, logo, Malformed, kitsch, extra legs,\
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
                Indistinct, Beyond the borders, text, Blurry,  \
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
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True, variant="fp16")
pipe = pipe.to("cuda")
pipe.load_lora_weights(inLoRA)
pipe.fuse_lora()
imgStages = []

if not os.path.isdir("genImg"):
    os.makedirs("genImg")

os.chdir("genImg")
dirAr = [x for x in os.listdir() if "run" in x]
outDir = "run"+str(len(dirAr)+1)
if not os.path.isdir(outDir):
    os.makedirs(outDir)

print(Fore.CYAN+Style.BRIGHT+"\n Latent imgOUT genImg\{} Dir Yo!".format(outDir)+Fore.RESET)

f = open(r"{}\prompt.txt".format(outDir), 'w')
inDict = {"guidanceScale-> " : gScale, "samplingSteps-> " : nSampling, "imgHeight-> " : h, "imgWidth-> " : w, 
          "seedVal-> " : valSeed, "Prompt-> " : pPrompt, "negativePrompt-> " : nPrompt}
inTxt = '\n'.join(y+str(inDict[y]).replace('  ','') for y in inDict)
f.write(inTxt)
f.close()

def callbackLatent(pipe, i, t, latents):
    latents_ = 1 / 0.18215 * latents['latents']
    with torch.no_grad():
        imgStage = pipe.vae.decode(latents_).sample[0]
    imgStage = (imgStage / 2 + 0.5).clamp(0, 1)
    imgStage = imgStage.cpu().permute(1, 2, 0).numpy()
    imgStages.extend(pipe.numpy_to_pil(imgStage))
    return latents

t1 = time.time()
img = pipe(prompt = pPrompt, negative_prompt = nPrompt, height = h, 
        width = w, num_inference_steps = nSampling, generator = gSeed, 
        guidance_scale = gScale, callback_on_step_end = callbackLatent).images[0]
imgStages.append(img)

for i, img in tqdm(enumerate(imgStages), desc="Gettin' images Yo!", colour="red"):
    img.save(outDir+"\{}.png".format(str(i)+'_'+str(time.time())))
t2 = time.time()

print("\nExecTime: ", (t2-t1))