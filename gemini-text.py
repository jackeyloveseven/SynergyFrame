from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from rembg import remove, new_session
import torch
from ip_adapter import IPAdapterXL
from PIL import Image, ImageChops
import numpy as np
import cv2
import os
from depth_anything_v2.dpt import DepthAnythingV2
from ip_adapter.Geometry_Estimating import MultiScaleDepthEnhancement, DirectionalShadingModule
# GEMINI REFACTOR: Import the custom attention processor from its own file
from attention_processor import CrossAttentionSwapProcessor

obj = '2'
texture = 'cup_glaze'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Setup Models (Depth, ControlNet, SDXL, IP-Adapter) ---
model_configs = {
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
}
depth_anything = DepthAnythingV2(**model_configs['vitb'])
depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
depth_anything = depth_anything.to(device).eval()

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"

torch.cuda.empty_cache()

controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    use_safetensors=True,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)

ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# --- 2. Prepare Input Images (Content, Mask, Depth, Shading) ---
target_image_path = 'demo_assets/input_imgs/' + obj + '.png'
raw_image_bgr = cv2.imread(target_image_path)
depth = depth_anything.infer_image(raw_image_bgr, 518)
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth = depth.astype(np.float32)
depth_3c = np.repeat(depth[..., np.newaxis], 3, axis=-1)

msdem = MultiScaleDepthEnhancement(edge_low_threshold=50, edge_high_threshold=150, feature_weights=(0.01, 0.01, 0.1))
enhanced_depth = msdem.enhance(depth_3c, raw_image_bgr)

target_image = Image.open(target_image_path).convert('RGB')
rm_bg = remove(target_image, session=new_session("isnet-general-use"))
alpha_channel = rm_bg.getchannel('A')
target_mask = alpha_channel.point(lambda p: 255 if p > 128 else 0).convert('RGB')

dsm = DirectionalShadingModule(ambient_strength=0.7, diffuse_strength=0.9)
light_dir = [-0.7, 0.3, 0.5] # Dramatic lighting
init_img_np = dsm.simulate_lighting(np.array(target_image), enhanced_depth, light_dir, np.array(target_mask))
init_img = Image.fromarray(init_img_np)

ip_image = Image.open("demo_assets/material_exemplars/" + texture + ".png")
depth_map = Image.fromarray(enhanced_depth).resize((1024,1024))
init_img = init_img.resize((1024,1024))
mask = target_mask.resize((1024, 1024))

# --- 3. Textual Inversion and Prompt Engineering ---
# pipe.load_textual_inversion("sd-concepts-library/walter-wick-photography")
pipe.load_textual_inversion("sd-concepts-library/fp-hk-photo-online", mean_resizing=False)
# pipe.load_textual_inversion("sd-concepts-library/midjourney-style") # midjourney
# pipe.load_textual_inversion("sd-concepts-library/line-art")
# pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons") # simplist
# pipe.load_textual_inversion("sd-concepts-library/moebius") # strange

structure_prompt = "realistic, masterpiece, high quality, high resolution, 8k"
texture_prompt = "vase texture, chinese porcelain"
negative_prompt = "low quality, unrealistic, blurry, deformed"

# --- 4. Advanced Control: Cross-Attention Swapping ---
(prompt_embeds_structure, negative_prompt_embeds, 
 pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(structure_prompt, negative_prompt=negative_prompt)

(prompt_embeds_texture, _, _, _) = pipe.encode_prompt(texture_prompt, num_images_per_prompt=1, do_classifier_free_guidance=False)

prompt_embeds = torch.cat([prompt_embeds_structure, prompt_embeds_texture], dim=1)
negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds], dim=1)
texture_prompt_seq_len = prompt_embeds_texture.shape[1]

swap_processor = CrossAttentionSwapProcessor(texture_prompt_seq_len)
print("Injecting custom Cross-Attention Swap Processor...")

attn_processors_backup = pipe.unet.attn_processors
pipe.unet.set_attn_processor(swap_processor)

# --- 5. Generation ---
# Get both conditional and unconditional IP-Adapter embeddings for CFG
cond_ip_embeds, uncond_ip_embeds = ip_model.get_image_embeds(ip_image)
# Concatenate them in the order [unconditional, conditional]
ip_adapter_image_embeds = torch.cat([uncond_ip_embeds, cond_ip_embeds], dim=0)

print("Starting generation with attention swapping...")
images = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    image=init_img,
    control_image=depth_map,
    mask_image=mask,
    controlnet_conditioning_scale=0.8,
    num_inference_steps=30,
    seed=42,
    # GEMINI FIX: The pipeline expects a list of tensors for this argument.
    ip_adapter_image_embeds=[ip_adapter_image_embeds]
).images

# --- 6. Cleanup and Save ---
print("Restoring default attention processor...")
pipe.unet.set_attn_processor(attn_processors_backup)

images[0].save("output_kv_swapped.png")
print("Generation complete. Saved to output_kv_swapped.png")
