import glob
import cv2
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers.models.attention_processor import AttnProcessor2_0
import matplotlib
from rembg import remove, new_session
import torch
import torch.nn.functional as F
from ip_adapter import IPAdapterXL
from PIL import Image, ImageChops
import numpy as np
import os

from Geometry_Estimating import DirectionalShadingModule, MultiScaleDepthEnhancement
from depth_anything_v2.dpt import DepthAnythingV2

# --- 针对特定词的注意力蒙版处理器 (无变化) ---
class WordSwapAttnProcessor(AttnProcessor2_0):
    def __init__(self, mask_tensor, token_indices):
        super().__init__()
        self.mask = mask_tensor.unsqueeze(0).unsqueeze(0)
        self.token_indices = token_indices

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        if encoder_hidden_states is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)

        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if attention_probs.shape[-1] == encoder_hidden_states.shape[1]:
            h = w = int(np.sqrt(attention_probs.shape[1]))
            resized_mask = F.interpolate(
                self.mask.to(attention_probs.device, dtype=attention_probs.dtype),
                size=(h, w), mode='bilinear', align_corners=False
            ).squeeze().view(1, -1, 1)
            
            modulation_matrix = torch.ones_like(attention_probs)
            for token_idx in self.token_indices:
                modulation_matrix[:, :, token_idx] = resized_mask.squeeze(-1)

            attention_probs = attention_probs * modulation_matrix
            attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-9)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

# --- 主程序 ---
obj = '5'
textual_inversion_id = "sd-concepts-library/doose-s-realistic-art-style"
placeholder_token = "<doose-s-realistic-art-style>"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        feature_weights=(0.008, 0.008, 0.6)
    )

    # 一行代码完成增强
    enhanced_depth = msdem.enhance(depth, raw_image)

    cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(filename))[0] + '.png'), enhanced_depth)




base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"



torch.cuda.empty_cache()
controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, use_safetensors=True,
    torch_dtype=torch.float16, add_watermarker=False
).to(device)

pipe.load_textual_inversion(textual_inversion_id)
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)







mask_source_image = Image.open(f'demo_assets/input_imgs/{obj}.png').convert('RGB')
target_mask = remove(mask_source_image, session=new_session("isnet-general-use")).getchannel('A')

prompt = f" blue house ."

# 直接指定固定的token_indices，通常风格词会在句子后半部分
token_indices = [1, 2, 3]  # 固定的token位置，这里设置为可能包含style描述的位置

# 打印一下当前使用的触发词和索引
print(f"使用触发词: '{placeholder_token}' 固定索引位置: {token_indices}")

# 准备光照和初始图像 (这部分逻辑保持不变)
# ... (你的光照模拟代码) ...
# 预定义8个光照方向
LIGHT_DIRECTIONS = {
    'top': [0.0, 0.0, 1.0], 'top_left': [-0.5, -0.5, 1.0], 'top_right': [0.5, -0.5, 1.0],
    'left': [-1.0, 0.0, 0.5], 'right': [1.0, 0.0, 0.5], 'front': [0.0, 1.0, 0.5],
    'front_top': [0.0, 0.5, 1.0], 'dramatic': [-0.7, 0.3, 0.5],
}
dsm = DirectionalShadingModule(ambient_strength=0.3, diffuse_strength=0.7)
target_image_np = np.array(mask_source_image)
# 假设depth和mask已经准备好
depth_np = np.array(enhanced_depth)
mask_np = np.array(target_mask)
light_dir = LIGHT_DIRECTIONS['right']
init_img_np = dsm.simulate_lighting(target_image_np, depth_np, light_dir, mask_np)
init_img = Image.fromarray(init_img_np)




# if not token_indices:
#     raise ValueError(f"触发词 '{placeholder_token}' 未在prompt中找到!")
print(f"成功定位触发词 '{placeholder_token}' 的索引: {token_indices}")

mask_tensor = torch.from_numpy(np.array(target_mask)).float() / 255.0
word_swap_proc = WordSwapAttnProcessor(mask_tensor=mask_tensor, token_indices=token_indices)

# --- 代码修正处 ---
# 4. 将处理器应用到U-Net的所有注意力层
attn_procs = {}
# 实例化一个默认处理器以用于我们不想修改的层
default_proc = AttnProcessor2_0()
for name in pipe.unet.attn_processors.keys():
    if name.endswith("attn2.processor"): # 如果是交叉注意力层
        attn_procs[name] = word_swap_proc # 使用我们自定义的处理器
    else: # 否则 (即自注意力层)
        attn_procs[name] = default_proc # 使用默认处理器
pipe.unet.set_attn_processor(attn_procs)
# --- 修正结束 ---

init_img = mask_source_image.resize((1024, 1024))
depth_map = Image.open(f'demo_assets/depths/{obj}.png').resize((1024, 1024))
pil_mask = target_mask.resize((1024, 1024), Image.LANCZOS)

print("开始 Image-to-Image 生成 (Textual Inversion控制风格)... ")
images = ip_model.generate(
    prompt=prompt,
    scale=0.01,
    pil_image=Image.new("RGB", (1024, 1024)),
    image=init_img,
    mask_image=pil_mask,
    control_image=depth_map,
    controlnet_conditioning_scale=0.9,
    num_samples=1,
    num_inference_steps=30,
    seed=42
)

images[0].save("output_text_controlled_style.png")

print("\n生成完成！结果已保存到 output_text_controlled_style.png")
print(f"Prompt: {prompt}")
