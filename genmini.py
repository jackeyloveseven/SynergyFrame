from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers.models.attention_processor import AttnProcessor2_0
from rembg import remove, new_session
import torch
import torch.nn.functional as F
from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from transformers import SamModel
import cv2
import glob
import matplotlib
import os
from depth_anything_v2.dpt import DepthAnythingV2
from Geometry_Estimating import MultiScaleDepthEnhancement, DirectionalShadingModule


# --- 新增：自定义注意力处理器 ---
class MaskedStyleAttnProcessor(AttnProcessor2_0):
    """
    自定义注意力处理器，用于根据蒙版控制IP-Adapter的风格应用范围。
    它通过在计算注意力权重后，使用一个二值蒙版来调制（multiply）注意力图，
    从而将风格注入限制在蒙版指定的区域内。
    """
    def __init__(self, mask_tensor=None):
        super().__init__()
        if mask_tensor is not None:
            # 为插值做准备，增加batch和channel维度
            self.mask = mask_tensor.unsqueeze(0).unsqueeze(0)
        else:
            self.mask = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        # 这是注意力模块的前向传播核心
        if self.mask is None:
            # 如果没有提供蒙版，则执行标准注意力操作
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)

        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # --- 核心修改：注入注意力蒙版 ---
        # 获取注意力图的空间维度 (h, w)
        h = w = int(np.sqrt(attention_probs.shape[1]))
        
        # 将我们的inpaint蒙版缩放到与注意力图相同的尺寸
        resized_mask = F.interpolate(
            self.mask.to(attention_probs.device, dtype=attention_probs.dtype),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )
        
        # 调整蒙版形状以匹配注意力图的形状 for broadcasting
        # (batch*heads, seq_len, kv_seq_len) -> we modulate seq_len
        resized_mask = resized_mask.squeeze().view(1, -1)
        resized_mask = resized_mask.repeat(attention_probs.shape[0], 1)
        
        # 用蒙版调制注意力图，`.unsqueeze(-1)`是为了广播到所有key
        attention_probs = attention_probs * resized_mask.unsqueeze(-1)
        
        # 重新归一化，确保权重和为1
        attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-9)
        # --- 修改结束 ---

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


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
    depth = depth.astype(np.float32)
    
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    
    enhanced_depth = depth

    # 创建增强器实例
    msdem = MultiScaleDepthEnhancement(
        edge_low_threshold=50,
        edge_high_threshold=150,
        feature_weights=(0.01, 0.01, 0.01)
    )

    # 一行代码完成增强
    enhanced_depth = msdem.enhance(depth, raw_image)

    cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(filename))[0] + '.png'), enhanced_depth)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"

torch.cuda.empty_cache()

# 加载SDXL管线
controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    use_safetensors=True,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)

# 准备输入图像和蒙版
target_image_path = 'demo_assets/input_imgs/' + obj + '.png'
target_image = Image.open(target_image_path).convert('RGB')
rm_bg = remove(target_image, session=new_session("isnet-general-use"))
target_mask = rm_bg.getchannel('A') # 从背景移除结果中获取Alpha通道作为蒙版

# --- 应用自定义注意力处理器 ---
# 1. 将PIL蒙版转换为tensor
mask_tensor = torch.from_numpy(np.array(target_mask)).float() / 255.0
# 2. 实例化我们的处理器
masked_attn_proc = MaskedStyleAttnProcessor(mask_tensor=mask_tensor)
# 3. 将处理器应用到U-Net的交叉注意力层
attn_procs = {}
for name in pipe.unet.attn_processors.keys():
    if name.endswith("attn2.processor"):
        attn_procs[name] = masked_attn_proc
    else:
        attn_procs[name] = pipe.unet.attn_processors[name]
pipe.unet.set_attn_processor(attn_procs)
# --- 设置完成 ---


# 正常加载IP-Adapter，它会与我们的自定义处理器协同工作
pipe.unet = register_cross_attention_hook(pipe.unet)
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"])

# 准备光照和初始图像 (这部分逻辑保持不变)
# ... (你的光照模拟代码) ...
# 预定义8个光照方向
LIGHT_DIRECTIONS = {
    'top': [0.0, 0.0, 1.0], 'top_left': [-0.5, -0.5, 1.0], 'top_right': [0.5, -0.5, 1.0],
    'left': [-1.0, 0.0, 0.5], 'right': [1.0, 0.0, 0.5], 'front': [0.0, 1.0, 0.5],
    'front_top': [0.0, 0.5, 1.0], 'dramatic': [-0.7, 0.3, 0.5],
}
dsm = DirectionalShadingModule(ambient_strength=0.9, diffuse_strength=1.9)
target_image_np = np.array(target_image)
# 假设depth和mask已经准备好
depth_np = np.array(enhanced_depth)
mask_np = np.array(target_mask)
light_dir = LIGHT_DIRECTIONS['right']
init_img_np = dsm.simulate_lighting(target_image_np, depth_np, light_dir, mask_np)
init_img = Image.fromarray(init_img_np)


# 准备其他输入
ip_image = Image.open("demo_assets/material_exemplars/" + texture + ".png")
depth_map = Image.open('demo_assets/depths/' + obj + '.png').resize((1024,1024))
init_img = init_img.resize((1024,1024))
# 注意：传递给管线的mask应该是PIL Image对象
pil_mask = target_mask.resize((1024, 1024), Image.LANCZOS)


# 可视化输入
grid = image_grid([target_image.resize((256, 256)), ip_image.resize((256, 256)), init_img.resize((256, 256)), depth_map.resize((256, 256))], 1, 4)
grid.save("input_grid_improved.png")

# 执行生成
num_samples = 3
# 注意这里的 `mask_image` 参数需要的是PIL格式的蒙版
images = ip_model.generate(
    pil_image=ip_image,
    image=init_img,
    control_image=depth_map,
    mask_image=pil_mask,
    controlnet_conditioning_scale=0.9,
    num_samples=num_samples,
    num_inference_steps=30,
    seed=42
)
images[0].save("output_improved.png")

print("改进后的生成完成，结果已保存到 output_improved.png")