from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from rembg import remove, new_session
import torch
from ip_adapter import IPAdapterXL,utils
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from transformers import SamModel
import cv2
import glob
import matplotlib
import os
from depth_anything_v2.dpt import DepthAnythingV2



obj = '5'
texture = 'cup_glaze'


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything = DepthAnythingV2(**model_configs['vitb'])
depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
depth_anything = depth_anything.to(device).eval()
img_path = 'demo_assets/input_imgs/'+obj+'.png'
out_dir = 'demo_assets/depths'
input_size = 518

if os.path.isfile(img_path):
    filenames = [img_path]
else:
    filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)

os.makedirs(out_dir, exist_ok=True)

cmap = matplotlib.colormaps.get_cmap('Spectral_r')


for k, filename in enumerate(filenames):
    print(f'Progress {k+1}/{len(filenames)}: {filename}')
    
    raw_image = cv2.imread(filename)
    
    depth = depth_anything.infer_image(raw_image, input_size)
    
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    print(depth.shape)

    cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)










def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"


torch.cuda.empty_cache()

# load SDXL pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    use_safetensors=True,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)
pipe.unet = register_cross_attention_hook(pipe.unet)

ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"])  # , target_blocks=["up_blocks.0.attentions.1"]



target_image_path = 'demo_assets/input_imgs/' + obj + '.png'  # Replace with your image path



target_image = Image.open(target_image_path).convert('RGB')
rm_bg = remove(target_image, session=new_session("isnet-general-use"))
# output.save(output_path)
target_mask = rm_bg.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB')# Convert mask to grayscale








# Ensure mask is the same size as image
# mask = ImageChops.invert(mask)
# Generate random noise for the size of the image
noise = np.random.randint(0, 256, target_image.size + (3,), dtype=np.uint8)
noise_image = Image.fromarray(noise)

print("target_mask.size:", target_mask.size)  # (W, H)
print("target_image.size:", target_image.size)  # (W, H)
mask_target_img = ImageChops.lighter(target_image, target_mask)
invert_target_mask = ImageChops.invert(target_mask)


gray_target_image = target_image.convert('L').convert('RGB')
gray_target_image = ImageEnhance.Brightness(gray_target_image)

# Adjust brightness
# The factor 1.0 means original brightness, greater than 1.0 makes the image brighter. Adjust this if the image is too dim
factor = 1.0  # Try adjusting this to get the desired brightness

gray_target_image = gray_target_image.enhance(factor)
grayscale_img = ImageChops.darker(gray_target_image, target_mask)
img_black_mask = ImageChops.darker(target_image, invert_target_mask)
grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
init_img = grayscale_init_img






ip_image = Image.open("demo_assets/material_exemplars/" + texture + ".png")

np_image = np.array(Image.open('demo_assets/depths/' + obj + '.png'))






depth_map = Image.fromarray(np_image).resize((1024,1024))

init_img = init_img.resize((1024,1024))
mask = target_mask.resize((1024, 1024))
mask.save("mask.png")
grid = image_grid([target_mask.resize((256, 256)), ip_image.resize((256, 256)), init_img.resize((256, 256)), depth_map.resize((256, 256))], 1, 4)

# Visualize each input individually
grid.save("input.png")
init_img.resize((256, 256)).save("init.png")


num_samples = 1
images = ip_model.generate(pil_image=ip_image, image=init_img, control_image=depth_map, mask_image=mask, controlnet_conditioning_scale=0.9, num_samples=num_samples,prompt='with text"Citizen"', num_inference_steps=30, seed=42)
images[0].save("output.png")