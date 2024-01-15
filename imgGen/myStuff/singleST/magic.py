from diffusers import DiffusionPipeline
from colorama import Fore, Style
from tkinter import filedialog
import torch
import tqdm
import time
import os

print(Fore.BLUE+Style.BRIGHT+"\n\nSelect the fineTuned SafeTensor File Yo!"+Fore.RESET)
inLoRA = filedialog.askopenfilename(filetypes=[("LoRA SafeTensor", "*.safetensors")])

w = 768
h = 768 
gScale = 7.5
valSeed = 42
nSampling = 28
pPrompt = "vSid a hyper-realistic man statue, best quality, upper body, \
    looking at viewer, simple background"
nPrompt = "low quality, ugly, tiling, poorly drawn hands, poorly drawn feet, \
    poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, \
        bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, \
            bad art, beginner, amateur, distorted face, blurry, draft, grain, worst quality, \
                bad anatomy, bad composition, poor, low effort"

gSeed = torch.Generator(device="cuda").manual_seed(valSeed)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = pipe.to("cuda")
pipe.load_lora_weights(inLoRA)
pipe.fuse_lora()

t1 = time.time()
img = pipe(prompt = pPrompt, negative_prompt = nPrompt, height = h, width = w,
           num_inference_steps = nSampling, generator = gSeed, guidance_scale = gScale).images[0]

if not os.path.isdir("genImg"):
   os.makedirs("genImg")
img.save("genImg\{}.png".format(time.time()))
t2 = time.time()

print("\nExecTime: ", (t2-t1))