from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel,  StableDiffusionXLControlNetImg2ImgPipeline
from rembg import remove, new_session
import torch
from ip_adapter.custom_ip_adapter2 import IPAdapterCustom
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from transformers import SamModel
import cv2
import glob
import matplotlib
import os
from depth_anything_v2.dpt import DepthAnythingV2
from Geometry_Estimating import MultiScaleDepthEnhancement, DirectionalShadingModule


obj = 'pinkwoman'
texture = 'cup_glaze'
texture = '5'


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
    
    enhanced_depth = depth

    # 创建增强器实例
    msdem = MultiScaleDepthEnhancement(
        edge_low_threshold=50,
        edge_high_threshold=150,
        featureweights=(1, 1, 1)
    )

    # 一行代码完成增强
    enhanced_depth = msdem.enhance(depth, raw_image)

    cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(filename))[0] + '.png'), enhanced_depth)










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
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    use_safetensors=True,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)
pipe.unet = register_cross_attention_hook(pipe.unet)

ip_model = IPAdapterCustom(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"], top_k=1)  # , target_blocks=["up_blocks.0.attentions.1"]



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




# 光照区
# 注释掉原生的灰度转换代码，因为我们现在使用新的光照模拟
# gray_target_image = target_image.convert('L').convert('RGB')
# gray_target_image = ImageEnhance.Brightness(gray_target_image)
# factor = 1.0
# gray_target_image = gray_target_image.enhance(factor)
# grayscale_img = ImageChops.darker(gray_target_image, target_mask)
# img_black_mask = ImageChops.darker(target_image, invert_target_mask)
# grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
# init_img = grayscale_init_img

# 预定义8个光照方向
LIGHT_DIRECTIONS = {
    'top': [0.0, 0.0, 1.0],        # 正上方
    'top_left': [-0.5, -0.5, 1.0],  # 左上方
    'top_right': [0.5, -0.5, 1.0],  # 右上方
    'left': [-1.0, 0.0, 0.5],       # 左侧
    'right': [1.0, 0.0, 0.5],       # 右侧
    'front': [0.0, 1.0, 0.5],       # 正面
    'front_top': [0.0, 0.5, 1.0],   # 正面偏上
    'dramatic': [-0.7, 0.3, 0.5],   # 戏剧性光照（左前上）
}

# 创建光照模拟模块实例
dsm = DirectionalShadingModule(
    ambient_strength=0.8,  # 环境光强度
    diffuse_strength=1.5   # 漫反射强度
)

# 将PIL图像转换为numpy数组
target_image_np = np.array(target_image)
depth_np = np.array(enhanced_depth)
mask_np = np.array(target_mask)

# 选择光照方向（这里可以根据需要更换标签）
light_direction = 'right'  # 可选: 'top', 'top_left', 'top_right', 'left', 'right', 'front', 'front_top', 'dramatic'
light_dir = LIGHT_DIRECTIONS[light_direction]

# 模拟新的光照方向
init_img_np = dsm.simulate_lighting(target_image_np, depth_np, light_dir, mask_np)
init_img = Image.fromarray(init_img_np)




ip_image = Image.open("demo_assets/material_exemplars/" + texture + ".png")

np_image = np.array(Image.open('demo_assets/depths/' + obj + '.png'))






depth_map = Image.fromarray(np_image).resize((1024,1024))

init_img = init_img.resize((1024,1024))
mask = target_mask.resize((1024, 1024))
grid = image_grid([target_mask.resize((256, 256)), ip_image.resize((256, 256)), init_img.resize((256, 256)), depth_map.resize((256, 256))], 1, 4)

# Visualize each input individually
grid.save("input.png")
init_img.resize((256, 256)).save("init.png")


num_samples = 3
images = ip_model.generate(pil_image=ip_image, image=init_img, control_image=depth_map, mask_image=mask, spatial_mask=target_mask, controlnet_conditioning_scale=0.9, num_samples=num_samples, num_inference_steps=30, seed=42)
images[0].save("custom_output2.png")
