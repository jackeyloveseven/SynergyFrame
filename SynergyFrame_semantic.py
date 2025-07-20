#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import cv2
import torch
import argparse
import warnings
import logging
from PIL import Image

# 只禁用特定警告，避免影响性能
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
warnings.filterwarnings("ignore", message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta")
warnings.filterwarnings("ignore", message="Anti-alias option is always applied for PIL Image input")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="`Transformer2DModelOutput` is deprecated")
warnings.filterwarnings("ignore", message="It seems like you have activated model offloading")

# 禁用OpenCV警告
logging.getLogger("cv2").setLevel(logging.ERROR)

from rembg import remove, new_session
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, AutoencoderKL, StableDiffusionXLControlNetPipeline
from transformers import SamModel, SamProcessor

# 导入自定义模块
from ip_adapter.custom_ip_adapter import IPAdapterCustom
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images
from depth_anything_v2.dpt import DepthAnythingV2
from Geometry_Estimating import MultiScaleDepthEnhancement, DirectionalShadingModule


def load_config(config_path='config.json'):
    """
    从JSON配置文件加载配置
    
    参数:
        config_path: 配置文件路径
        
    返回:
        config_dict: 配置字典
    """
    default_config = {
        # 输入/输出路径
        'obj': '5',
        'texture': '5',
        'input_dir': 'demo_assets/input_imgs/',
        'texture_dir': 'demo_assets/material_exemplars/',
        'depth_dir': 'demo_assets/depths',
        'output_file': 'synergy_output.png',
        
        # 光照配置
        'light_direction': 'right',
        'ambient_strength': 0.8,
        'diffuse_strength': 1.5,
        
        # CUDA加速配置
        'use_cuda': True,  # 是否使用CUDA
        'use_mixed_precision': False,  # 是否使用混合精度
        'use_fp16': False,  # 是否使用半精度
        'use_xformers': True  # 是否使用xformers内存优化
    }
    
    # 尝试从文件加载配置
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                # 更新默认配置
                default_config.update(file_config)
            # print(f"已从 {config_path} 加载配置")
        except Exception as e:
            print(f"加载配置文件 {config_path} 时出错: {str(e)}，将使用默认配置")
    else:
        print(f"配置文件 {config_path} 不存在，将使用默认配置")
        
    return default_config

def parse_args(config_dict=None):
    """
    解析命令行参数
    
    参数:
        config_dict: 从配置文件加载的配置字典
        
    返回:
        args: 解析后的参数命名空间
    """
    if config_dict is None:
        config_dict = load_config()
        
    parser = argparse.ArgumentParser(description='SynergyFrame: 材质与物体融合系统')
    
    # 添加配置文件参数
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    
    # 输入/输出路径
    parser.add_argument('--obj', type=str, default=config_dict['obj'], help='目标物体图片名称或编号')
    parser.add_argument('--texture', type=str, default=config_dict['texture'], help='材质图片名称或编号')
    parser.add_argument('--input_dir', type=str, default=config_dict['input_dir'], help='输入图像目录')
    parser.add_argument('--texture_dir', type=str, default=config_dict['texture_dir'], help='材质示例目录')
    parser.add_argument('--depth_dir', type=str, default=config_dict['depth_dir'], help='深度图输出目录')
    parser.add_argument('--output_file', type=str, default=config_dict['output_file'], help='输出图像文件名')
    
    # 光照配置
    parser.add_argument('--light_direction', type=str, default=config_dict['light_direction'], 
                        choices=['top', 'top_left', 'top_right', 'left', 'right', 'front', 'front_top', 'dramatic'], 
                        help='光照方向')
    parser.add_argument('--ambient_strength', type=float, default=config_dict['ambient_strength'], help='环境光强度')
    parser.add_argument('--diffuse_strength', type=float, default=config_dict['diffuse_strength'], help='漫反射强度')
    
    parser.add_argument('--sam', type=bool, default=config_dict.get('sam', False), help='是否使用SAM模型')
    parser.add_argument('--backbone', type=str, default=config_dict.get('backbone', 'Inpaint'), choices=['Img2Img', 'Inpaint'])
    
    # 添加手动蒙版相关参数
    parser.add_argument('--use_manual_mask', type=bool, default=config_dict.get('use_manual_mask', False), help='是否使用手动绘制的蒙版')
    parser.add_argument('--mask_path', type=str, default=config_dict.get('mask_path', ''), help='手动绘制蒙版的路径')
    
    return parser.parse_args()


class DepthEstimator:
    """深度估计模块"""
    
    # 模型配置字典
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    def __init__(self, model_type, checkpoint_path, device, featureweights, input_size=518):
        """
        初始化深度估计器
        
        参数:
            model_type: 模型类型 ('vits', 'vitb', 'vitl', 'vitg')
            checkpoint_path: 检查点路径
            device: 运行设备
            input_size: 输入图像大小
        """
        self.device = device
        self.input_size = input_size
        
        # 加载深度估计模型
        self.depth_model = DepthAnythingV2(**self.MODEL_CONFIGS[model_type])
        self.depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        self.depth_model = self.depth_model.to(device).eval()
        
        # 初始化深度增强器
        self.depth_enhancer = MultiScaleDepthEnhancement(
            edge_low_threshold=50,
            edge_high_threshold=150,
            featureweights=featureweights
        )
        
    def estimate_depth(self, image_path, output_dir):
        """
        估计图像深度并保存
        
        参数:
            image_path: 输入图像路径
            output_dir: 输出目录
            
        返回:
            enhanced_depth: 增强后的深度图像
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载图像
        raw_image = cv2.imread(image_path)
        
        # 估计深度
        depth = self.depth_model.infer_image(raw_image, self.input_size)
        
        # 归一化深度
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        
        # 增强深度
        enhanced_depth = self.depth_enhancer.enhance(depth, raw_image)
        
        # 保存深度图
        output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.png')
        cv2.imwrite(output_filename, enhanced_depth)
        
        return enhanced_depth


class LightingSimulator:
    """光照模拟模块"""
    
    # 预定义光照方向
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
    
    def __init__(self, ambient_strength=0.8, diffuse_strength=1.5):
        """
        初始化光照模拟器
        
        参数:
            ambient_strength: 环境光强度
            diffuse_strength: 漫反射强度
        """
        self.shading_module = DirectionalShadingModule(
            ambient_strength=ambient_strength,
            diffuse_strength=diffuse_strength
        )
    
    def simulate_lighting(self, image, depth, mask, direction='right'):
        """
        模拟光照效果
        
        参数:
            image: 原始图像 (PIL.Image)
            depth: 深度图 (numpy.ndarray)
            mask: 物体遮罩 (PIL.Image)
            direction: 光照方向
            
        返回:
            lit_image: 应用光照后的图像 (PIL.Image)
        """
        # 转换为numpy数组
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # 获取光照方向向量
        light_dir = self.LIGHT_DIRECTIONS[direction]
        
        # 应用光照模拟
        lit_image_np = self.shading_module.simulate_lighting(
            image_np, depth, light_dir, mask_np
        )
        
        # 转换回PIL图像
        lit_image = Image.fromarray(lit_image_np)
        
        return lit_image


class ImageProcessor:
    """图像处理模块"""
    
    def __init__(self):
        """初始化图像处理器"""
        self.rembg_session = new_session("isnet-general-use")
    
    def remove_background(self, image):
        """
        移除图像背景
        
        参数:
            image: 输入图像 (PIL.Image)
            
        返回:
            mask: 物体遮罩 (PIL.Image)
        """
        # 移除背景
        rm_bg = remove(image, session=self.rembg_session)
        
        # 创建遮罩
        mask = rm_bg.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB')
        
        return mask
    
    def remove_background_with_sam(self, image):
        """
        使用SAM模型移除图像背景，采用automatic mask生成方式
        
        参数:
            image: 输入图像 (PIL.Image)
            
        返回:
            mask: 物体遮罩 (PIL.Image)
        """
        # 初始化SAM模型（可以在类初始化时完成）
        if not hasattr(self, 'sam_model'):
            self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to('cuda' if torch.cuda.is_available() else 'cpu')
            self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        
        # 获取图像尺寸
        width, height = image.size
        
        # 自动生成中心点作为提示点
        center_point = [[[width // 2, height // 2]]]  # 图像中心点
        
        # 处理图像并生成遮罩
        inputs = self.sam_processor(image, input_points=center_point, return_tensors="pt").to(self.sam_model.device)
        
        # 运行模型
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        # 后处理获取遮罩
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # 获取IoU分数
        scores = outputs.iou_scores.cpu().numpy()
        
        # 选择最佳遮罩（具有最高IoU分数的掩码）
        best_mask_idx = np.argmax(scores[0][0])
        mask = masks[0][0][best_mask_idx].numpy()
        
        # 转换为PIL图像并返回
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB')
        return mask_image

    def donot_remove(self, image):
        """
        不移除背景，创建全白蒙版
        """
        # 创建与图像大小相同的全白图像
        width, height = image.size
        white_array = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 使用与其他蒙版处理一致的方式处理全白蒙版
        mask_image = Image.fromarray(white_array).convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB')
        return mask_image
    
    def create_image_grid(self, images, rows, cols):
        """
        创建图像网格
        
        参数:
            images: 图像列表
            rows: 行数
            cols: 列数
            
        返回:
            grid: 网格图像
        """
        assert len(images) == rows * cols
        
        w, h = images[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        
        for i, img in enumerate(images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
            
        return grid


def main():
    """主函数"""
    # 内部配置参数
    # 模型配置
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    image_encoder_path = "models/image_encoder"
    ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
    controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
    
    #深度模型配置
    depth_model = "vitb"
    depth_ckpt = "checkpoints/depth_anything_v2_vitb.pth"
    input_size = 518
    # CUDA配置 - 这些将被命令行或配置文件的值覆盖
    
    # 解析命令行参数（首先获取配置文件路径）
    temp_args, _ = argparse.ArgumentParser().parse_known_args()
    config_path = getattr(temp_args, 'config', 'config.json')
    
    # 从配置文件加载配置
    config_dict = load_config(config_path)
    
    # 解析命令行参数（命令行参数优先级高于配置文件）
    args = parse_args(config_dict)
    args.sam = config_dict.get('sam', False)
    args.use_manual_mask = config_dict.get('use_manual_mask', False)
    args.mask_path = config_dict.get('mask_path', '')
    args.donot_remove = config_dict.get('donot_remove', False)
    # 从config_dict直接读取CUDA相关参数
    args.use_cuda = config_dict.get('use_cuda', True)
    args.use_mixed_precision = config_dict.get('use_mixed_precision', True)
    args.use_fp16 = config_dict.get('use_fp16', True)
    args.use_xformers = config_dict.get('use_xformers', True)
    featureweights = [config_dict.get('featureweights1', 1), config_dict.get('featureweights2', 1), config_dict.get('featureweights3', 1)]
    # 生成配置
    num_samples = config_dict.get('num_samples', 1)
    num_steps = config_dict.get('num_steps', 30)
    seed = config_dict.get('seed', 3704)
    controlnet_scale = config_dict.get('controlnet_scale', 0.9)
    # 设置设备
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
        # print(f"使用设备: {device}")
        # print(f"CUDA版本: {torch.version.cuda}")
        # print(f"可用GPU: {torch.cuda.get_device_name(0)}")
        # print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = 'mps'
        # print(f"使用设备: {device} (Apple Metal)")
    else:
        device = 'cpu'
        # print(f"使用设备: {device} (没有找到可用的GPU)")
        
    # # 打印加速设置
    # if device == 'cuda':
    #     print(f"混合精度加速: {'启用' if args.use_mixed_precision else '禁用'}")
    #     print(f"半精度计算(FP16): {'启用' if args.use_fp16 else '禁用'}")
    #     print(f"Xformers内存优化: {'启用' if args.use_xformers else '禁用'}")
    
    # 构建文件路径
    for file in os.listdir(args.input_dir):
        if file.startswith(args.obj):
            obj_image_path = os.path.join(args.input_dir, file)
            break
    for file in os.listdir(args.texture_dir):
        if file.startswith(args.texture):
            texture_image_path = os.path.join(args.texture_dir, file)
            break


    # 初始化深度估计器
    depth_estimator = DepthEstimator(
        depth_model, 
        depth_ckpt, 
        device, 
        input_size=input_size,
        featureweights=featureweights
    )
    
    # 估计深度
    enhanced_depth = depth_estimator.estimate_depth(obj_image_path, args.depth_dir)
    
    # 初始化图像处理器
    image_processor = ImageProcessor()
    
    # 加载目标图像并移除背景
    target_image = Image.open(obj_image_path).convert('RGB')
    
    # 保存原始图像尺寸，用于后续还原
    original_width, original_height = target_image.size
    original_aspect_ratio = original_width / original_height
    
    # 根据原始宽高比计算新的尺寸，保持长边为1024
    if original_width >= original_height:
        # 宽边是长边
        target_width = 1024
        target_height = int(1024 / original_aspect_ratio)
    else:
        # 高边是长边
        target_height = 1024
        target_width = int(1024 * original_aspect_ratio)
    
    # # 打印原始尺寸和目标尺寸
    # print(f"原始图像尺寸: {original_width}x{original_height}, 宽高比: {original_aspect_ratio:.2f}")
    # print(f"目标输出尺寸: {target_width}x{target_height}")
    
    # 处理蒙版：优先使用手动蒙版，其次使用SAM或自动分割
    if args.donot_remove:
        print("不移除背景")
        target_mask = image_processor.donot_remove(target_image)
    elif args.use_manual_mask and os.path.exists(args.mask_path):
        print(f"使用手动绘制的蒙版: {args.mask_path}")
        target_mask = Image.open(args.mask_path).convert('RGB')
    elif args.sam:
        print("使用SAM进行分割")
        target_mask = image_processor.remove_background_with_sam(target_image)
    else:
        print("使用自动分割")
        target_mask = image_processor.remove_background(target_image)
    
    # 初始化光照模拟器
    lighting_simulator = LightingSimulator(
        ambient_strength=args.ambient_strength,
        diffuse_strength=args.diffuse_strength
    )
    
    # 应用光照模拟
    init_img = lighting_simulator.simulate_lighting(
        target_image, 
        enhanced_depth, 
        target_mask, 
        args.light_direction
    )
    
    # 加载材质图像
    texture_image = Image.open(texture_image_path)
    
    # 直接使用enhanced_depth，避免重新打开文件
    depth_map = Image.fromarray(enhanced_depth).resize((1024, 1024))
    
    # 调整图像大小
    init_img = init_img.resize((1024, 1024))
    init_img.save("init_img.png")
    mask = target_mask.resize((1024, 1024))
    mask.save("mask.png")
    # 创建输入预览
    grid = image_processor.create_image_grid(
        [
            target_mask.resize((256, 256)), 
            texture_image.resize((256, 256)), 
            init_img.resize((256, 256)), 
            depth_map.resize((256, 256))
        ], 
        1, 
        4
    )
    grid.save("input_preview.png")
    
    
    # 清空CUDA缓存
    if device == 'cuda':
        torch.cuda.empty_cache()
        
    # 设置数据类型
    torch_dtype = torch.float16 if args.use_fp16 and device == 'cuda' else torch.float32
    variant = "fp16" if args.use_fp16 and device == 'cuda' else None
    
    # 加载ControlNet模型
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, 
        variant=variant, 
        use_safetensors=True, 
        torch_dtype=torch_dtype
    ).to(device)
    
    choice_backbone = {
        'Img2Img': StableDiffusionXLControlNetImg2ImgPipeline,
        'Inpaint': StableDiffusionXLControlNetInpaintPipeline,
        'T2I': StableDiffusionXLControlNetPipeline
    }


    # 加载SDXL管道
    pipe = choice_backbone[args.backbone].from_pretrained(
        base_model_path,
        controlnet=controlnet,
        use_safetensors=True,
        torch_dtype=torch_dtype,
        add_watermarker=False,
    ).to(device)


    # 启用内存优化
    if device == 'cuda':
        if args.use_xformers:
            try:
                # 尝试启用xformers加速
                pipe.enable_xformers_memory_efficient_attention()
                # print("已启用xformers内存优化")
            except Exception as e:
                print(f"无法启用xformers: {str(e)}")
                print("请确保已安装xformers: pip install xformers")
        
        # 使用模型CPU卸载而不是手动移动到GPU
        # 注释掉下面这行可以避免警告
        # pipe.enable_model_cpu_offload()
        # if args.use_mixed_precision:
        #     print("已启用混合精度计算")
    
    # 注册交叉注意力钩子
    pipe.unet = register_cross_attention_hook(pipe.unet)
    
    # 初始化IP适配器
    ip_model = IPAdapterCustom(
        pipe, 
        image_encoder_path, 
        ip_ckpt, 
        device, 
        target_blocks=["down_blocks.2.attentions.1","down_blocks.3.attentions.0","up_blocks.0.attentions.1","up_blocks.1.attentions.0"]
    )
    # 语义："down_blocks.2.attentions.1","down_blocks.3.attentions.0"
    # 纹理："up_blocks.0.attentions.1","up_blocks.1.attentions.0"

    # 生成图像
    images = ip_model.generate(
        pil_image=texture_image,
        image=init_img,
        control_image=depth_map,
        mask_image=mask,
        spatial_mask=target_mask,
        controlnet_conditioning_scale=controlnet_scale,
        num_samples=num_samples,
        num_inference_steps=num_steps,
        seed=seed,
        scale=config_dict.get('scale', 1.0),
        prompt=config_dict.get('prompt'),
        semantic_scale=config_dict.get('semantic_scale', 0.5)
    )
    
    # 将生成的图像调整为原始图像的宽高比
    output_image = images[0].resize((target_width, target_height), Image.LANCZOS)
    
    # 保存输出
    output_image.save(args.output_file)
    


if __name__ == "__main__":
    main() 