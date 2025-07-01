from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers.models.attention_processor import AttnProcessor2_0
from rembg import remove, new_session
import torch
import torch.nn.functional as F
from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook
from PIL import Image, ImageChops
import numpy as np
import os
from typing import Dict, List

# --- 注意力处理器 1: 蒙版化风格应用 (用于交叉注意力) ---
class MaskedStyleAttnProcessor(AttnProcessor2_0):
    def __init__(self, mask_tensor=None):
        super().__init__()
        self.mask = mask_tensor.unsqueeze(0).unsqueeze(0) if mask_tensor is not None else None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        if self.mask is None or encoder_hidden_states is None: # 只在交叉注意力上生效
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)

        # ... (标准注意力计算前置步骤)
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

        # --- 核心修改：用蒙版调制注意力图 ---
        h = w = int(np.sqrt(attention_probs.shape[1]))
        resized_mask = F.interpolate(self.mask.to(attention_probs.device, dtype=attention_probs.dtype), size=(h, w), mode='bilinear', align_corners=False)
        resized_mask = resized_mask.squeeze().view(1, -1).repeat(attention_probs.shape[0], 1)
        attention_probs = attention_probs * resized_mask.unsqueeze(-1)
        attention_probs = attention_probs / (attention_probs.sum(dim=-1, keepdim=True) + 1e-9)
        # --- 修改结束 ---

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

# --- 注意力处理器 2: 纹理结构注入 (用于自注意力) ---
class TextureStructureInjectionAttnProcessor(AttnProcessor2_0):
    def __init__(self, kv_cache: Dict[str, torch.Tensor], scale: float = 1.0):
        super().__init__()
        self.kv_cache = kv_cache
        self.scale = scale

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        # 如果是交叉注意力，则跳过
        if encoder_hidden_states is not None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb, scale)

        # --- 核心修改：替换K和V向量 ---
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        
        # 从缓存中获取预先计算好的K和V
        ref_key = self.kv_cache["key"]
        ref_value = self.kv_cache["value"]
        
        # 调整K和V的尺寸以匹配当前query
        # (这在U-Net的不同层级是必要的)
        if query.shape[1] != ref_key.shape[1]:
            ref_key = F.interpolate(ref_key.permute(0, 2, 1), size=(query.shape[1],), mode='linear').permute(0, 2, 1)
            ref_value = F.interpolate(ref_value.permute(0, 2, 1), size=(query.shape[1],), mode='linear').permute(0, 2, 1)

        key = ref_key.to(query.device, dtype=query.dtype)
        value = ref_value.to(query.device, dtype=query.dtype)
        
        # (可选) 与原始K,V进行混合，以控制注入强度
        if self.scale < 1.0:
            original_key = attn.to_k(hidden_states)
            original_value = attn.to_v(hidden_states)
            key = (1 - self.scale) * original_key + self.scale * key
            value = (1 - self.scale) * original_value + self.scale * value
        # --- 修改结束 ---

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
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

# --- 辅助函数：提取自注意力K/V缓存 ---
def extract_self_attention_kv(pipe, texture_image, target_size=(1024, 1024)):
    print("正在从纹理图中提取结构K/V缓存...")
    kv_cache = {}
    hooks = []

    def hook_fn(module, args, kwargs, output):
        # args[0] is hidden_states
        hidden_states = args[0]
        # 在AttnProcessor内部计算K和V
        key = module.to_k(hidden_states)
        value = module.to_v(hidden_states)
        kv_cache[module.name] = {"key": key.cpu(), "value": value.cpu()}

    # 只在U-Net最深的自注意力层挂钩子以获取最关键的结构信息
    target_block_name = "down_blocks.2.attentions.0.processor.to_k"
    for name, module in pipe.unet.named_modules():
        if name.endswith("attn1.processor"): # attn1是自注意力
            module.name = name # 动态添加名字属性以便在hook中识别
            hooks.append(module.register_forward_hook(hook_fn, with_kwargs=True))
    
    # 预处理纹理图并送入U-Net
    image_tensor = pipe.image_processor.preprocess(texture_image.resize(target_size))
    latents = pipe.vae.encode(image_tensor.to(device, dtype=pipe.vae.dtype)).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    t = torch.tensor([pipe.scheduler.timesteps[0]], device=device)

    # === 新增：准备text_embeds和time_ids ===
    prompt = ""  # 用空字符串即可
    negative_prompt = ""
    text_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=negative_prompt
    )
    original_size = target_size
    crop_coords = (0, 0, target_size[0], target_size[1])
    time_ids = pipe._get_add_time_ids(
        original_size, crop_coords, target_size, device
    )
    added_cond_kwargs = {
        "text_embeds": text_embeds,
        "time_ids": time_ids
    }
    # 只需通过U-Net一次即可触发hook
    with torch.no_grad():
        pipe.unet(latents, t, encoder_hidden_states=None, added_cond_kwargs=added_cond_kwargs)

    # 移除钩子
    for h in hooks: h.remove()
    print("K/V缓存提取完成。")
    return kv_cache

# --- 主程序 ---
# 参数设置
obj = '5'
texture = 'cup_glaze'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 可调参数 ---
ip_scale = 1.0  # IP-Adapter风格强度 (0.0 -> 1.5)
texture_injection_scale = 1.0 # 纹理结构注入强度 (0.0 -> 1.0, 1.0为完全替换)

# 模型路径
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"

# 加载管线
torch.cuda.empty_cache()
controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, use_safetensors=True,
    torch_dtype=torch.float16, add_watermarker=False
).to(device)

# 准备输入图像和蒙版
target_image = Image.open(f'demo_assets/input_imgs/{obj}.png').convert('RGB')
target_mask = remove(target_image, session=new_session("isnet-general-use")).getchannel('A')

# 准备纹理参考图
texture_image = Image.open(f"demo_assets/material_exemplars/{texture}.png").convert("RGB")

# --- 核心操作：设置双重注意力处理器 ---
# 1. 提取纹理结构的K/V缓存
kv_cache = extract_self_attention_kv(pipe, texture_image)

# 2. 准备蒙版张量
mask_tensor = torch.from_numpy(np.array(target_mask)).float() / 255.0

# 3. 实例化并应用两个处理器
attn_procs = {}
for name in pipe.unet.attn_processors.keys():
    if name.endswith("attn1.processor"): # 自注意力层
        # 从缓存中为当前层找到对应的K/V
        layer_kv_cache = kv_cache.get(name, None)
        if layer_kv_cache:
            attn_procs[name] = TextureStructureInjectionAttnProcessor(layer_kv_cache, scale=texture_injection_scale)
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    elif name.endswith("attn2.processor"): # 交叉注意力层
        attn_procs[name] = MaskedStyleAttnProcessor(mask_tensor=mask_tensor)
    else:
        attn_procs[name] = pipe.unet.attn_processors[name]
pipe.unet.set_attn_processor(attn_procs)

# 加载IP-Adapter并设置强度
pipe.unet = register_cross_attention_hook(pipe.unet)
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
ip_model.set_scale(ip_scale)

# 准备其他输入 (光照, ControlNet等)
# (为简化，此处省略光照和深度模拟代码，直接使用原图作为init_img)
init_img = target_image.resize((1024, 1024))
depth_map = Image.open(f'demo_assets/depths/{obj}.png').resize((1024,1024))
pil_mask = target_mask.resize((1024, 1024), Image.LANCZOS)

# 执行生成
print("开始最终生成...")
images = ip_model.generate(
    pil_image=texture_image,
    image=init_img,
    control_image=depth_map,
    mask_image=pil_mask,
    controlnet_conditioning_scale=0.8,
    num_samples=1,
    num_inference_steps=30,
    seed=42
)
images[0].save("output_final_enhanced.png")

print("\n生成完成！结果已保存到 output_final_enhanced.png")
print(f"当前设置: IP-Adapter强度={ip_scale}, 纹理结构注入强度={texture_injection_scale}")
