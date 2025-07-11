#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import numpy as np
from PIL import Image
import torch
import subprocess
from pathlib import Path
import time
import sys
import warnings
import cv2

# 忽略所有警告
warnings.filterwarnings("ignore")

# 导入Gradio - 不自动安装，确保提前手动安装好
import gradio as gr

# 获取当前目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载配置文件
def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 保存配置文件
def save_config(config_dict, config_path='config.json'):
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

# 运行SynergyFrame
def run_synergyframe(config_path='config.json'):
    subprocess.run([sys.executable,load_config(config_path).get('model'), "--config", config_path])
    output_file = load_config(config_path).get('output_file', 'synergy_output.png')
    return output_file, "input_preview.png"

# 获取所有物体图像选项
def get_object_options():
    input_dir = load_config().get('input_dir', 'data/content')
    objects = [os.path.basename(f) for f in glob.glob(f"{input_dir}/*") if os.path.isfile(f)]
    return sorted(objects)

# 获取所有材质纹理选项
def get_texture_options():
    texture_dir = load_config().get('texture_dir', 'data/style')
    textures = [os.path.basename(f) for f in glob.glob(f"{texture_dir}/*") if os.path.isfile(f)]
    return sorted(textures)

# 上传自定义图像并返回路径
def upload_custom_image(image, is_object=True):
    if image is None:
        return None
    
    # 创建临时目录(如果不存在)
    temp_dir = "temp_inputs" if is_object else "temp_textures"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 保存上传的图像
    filename = f"custom_{int(time.time())}.png"
    save_path = os.path.join(temp_dir, filename)
    
    try:
        # 将PIL图像转换为RGB并保存
        img = Image.open(image.name).convert('RGB')
        img.save(save_path)
        return save_path
    except Exception as e:
        print(f"处理上传图像时出错: {str(e)}")
        return None

# 处理标签切换 - 当选择预设时清除上传
def on_preset_tab_select():
    return None, None

# 处理标签切换 - 当选择上传时清除预设选择
def on_upload_tab_select():
    return gr.update(value=None), gr.update(value=None)

# 清除绘制并启用SAM智能分割
def clear_sketch_and_use_sam():
    return None, gr.update(value=True)

# 加载图像到绘图区域
def load_image_to_sketch(image_path):
    if image_path is None:
        return None
    try:
        if isinstance(image_path, str):
            return Image.open(image_path).convert('RGB')
        else:
            return image_path
    except Exception as e:
        print(f"加载图像到绘图区域时出错: {str(e)}")
        return None

# 处理图像选择变更
def on_image_change(image_path, is_preset=True):
    if image_path:
        if is_preset:
            # 预设图像
            input_dir = load_config().get('input_dir', 'data/content')
            full_path = os.path.join(input_dir, image_path)
            return load_image_to_sketch(full_path)
        else:
            # 上传的图像
            return load_image_to_sketch(image_path)
    return None

# 处理手动绘制的蒙版
def process_manual_mask(sketch_canvas):
    if sketch_canvas is None:
        return None, None
    
    # 创建临时目录(如果不存在)
    temp_dir = "temp_masks"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 保存手动绘制的蒙版
    mask_filename = f"manual_mask_{int(time.time())}.png"
    mask_path = os.path.join(temp_dir, mask_filename)
    
    try:
        # 将蒙版转换为二值图像
        mask_array = np.array(sketch_canvas["mask"])
        
        # 如果是彩色蒙版，转换为灰度
        if len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
        
        # 二值化
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        
        # 保存蒙版
        cv2.imwrite(mask_path, binary_mask)
        
        # 创建可视化预览
        # 获取原始图像
        original_image = np.array(sketch_canvas["image"])
        
        # 创建彩色蒙版用于预览
        color_mask = np.zeros_like(original_image)
        color_mask[:,:,1] = binary_mask  # 在绿色通道显示蒙版
        
        # 将蒙版叠加到原图上
        alpha = 0.5
        preview = cv2.addWeighted(original_image, 1, color_mask, alpha, 0)
        
        # 保存预览图像
        preview_path = os.path.join(temp_dir, f"preview_{int(time.time())}.png")
        cv2.imwrite(preview_path, preview)
        
        # 返回蒙版路径和预览图像
        return mask_path, Image.fromarray(preview)
    except Exception as e:
        print(f"处理手动蒙版时出错: {str(e)}")
        return None, None

# 使用SAM处理蒙版
def process_sam_with_points(image, points, labels):
    if image is None or not points:
        return None
    
    # 创建临时目录(如果不存在)
    temp_dir = "temp_sam_masks"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 保存点击点和标签
    points_filename = f"sam_points_{int(time.time())}.json"
    points_path = os.path.join(temp_dir, points_filename)
    
    # 保存图像的临时副本
    if isinstance(image, str):
        image_path = image
    else:
        # 如果是PIL图像或其他格式，保存为临时文件
        temp_image = f"temp_image_{int(time.time())}.png"
        image_path = os.path.join(temp_dir, temp_image)
        if isinstance(image, Image.Image):
            image.save(image_path)
        else:
            Image.fromarray(image).save(image_path)
    
    # 将点和标签保存为JSON
    with open(points_path, 'w') as f:
        json.dump({'points': points, 'labels': labels}, f)
    
    # 调用SAM处理脚本（这里假设您有一个单独的SAM处理脚本）
    # 在实际应用中，您可能需要直接在这里集成SAM模型的代码
    try:
        # 这里是一个示例，实际上您需要根据您的SAM实现来调整
        # subprocess.run([sys.executable, "sam_process.py", "--image", image_path, "--points", points_path])
        
        # 假设SAM处理后的蒙版保存在这个位置
        mask_path = os.path.join(temp_dir, f"sam_mask_{int(time.time())}.png")
        
        # 在这里，我们应该实际调用SAM模型
        # 由于这需要特定的SAM实现，我们这里只是创建一个示例蒙版
        # 在实际应用中，请替换为真正的SAM调用
        
        # 创建一个示例蒙版（仅用于演示）
        img = Image.open(image_path).convert('RGB')
        mask = Image.new('L', img.size, 0)
        for point, label in zip(points, labels):
            x, y = point
            # 如果标签为1，绘制白色圆圈（前景）
            if label == 1:
                cv2.circle(np.array(mask), (int(x), int(y)), 50, 255, -1)
        
        # 保存蒙版
        mask.save(mask_path)
        
        # 返回蒙版图像
        return Image.open(mask_path)
    except Exception as e:
        print(f"使用SAM处理蒙版时出错: {str(e)}")
        return None

# 根据UI输入更新配置
def update_config(obj, texture, backbone, light_direction, ambient_strength, diffuse_strength, 
                 use_sam, num_steps, controlnet_scale, prompt, seed, top_k,
                 feature_weight1, feature_weight2, feature_weight3,
                 custom_obj=None, custom_texture=None, sketch_canvas=None, scale=None):
    
    config = load_config()
    
    # 默认路径
    default_input_dir = 'data/content'
    default_texture_dir = 'data/texture'
    
    # 处理自定义图像
    if custom_obj is not None and hasattr(custom_obj, 'name'):
        # 使用自定义上传图像
        obj_path = upload_custom_image(custom_obj, is_object=True)
        if obj_path:
            config['input_dir'] = os.path.dirname(obj_path)
            config['obj'] = os.path.basename(obj_path).split('.')[0]
    elif obj:
        # 使用预设选择，恢复默认路径
        config['input_dir'] = default_input_dir
        config['obj'] = obj.split('.')[0]
    
    if custom_texture is not None and hasattr(custom_texture, 'name'):
        # 使用自定义上传纹理
        texture_path = upload_custom_image(custom_texture, is_object=False)
        if texture_path:
            config['texture_dir'] = os.path.dirname(texture_path)
            config['texture'] = os.path.basename(texture_path).split('.')[0]
    elif texture:
        # 使用预设选择，恢复默认路径
        config['texture_dir'] = default_texture_dir
        config['texture'] = texture.split('.')[0]
    
    # 处理手动蒙版
    if sketch_canvas is not None and "mask" in sketch_canvas:
        # 检查蒙版是否为空（全黑）
        is_empty_mask = True
        if "mask" in sketch_canvas and sketch_canvas["mask"] is not None:
            mask_array = np.array(sketch_canvas["mask"])
            # 检查蒙版是否包含非零像素
            if np.any(mask_array > 0):
                is_empty_mask = False
        
        if not is_empty_mask:
            # 处理手动绘制的蒙版
            mask_path, _ = process_manual_mask(sketch_canvas)
            if mask_path:
                # 更新配置以使用手动蒙版
                config['mask_path'] = mask_path
                config['use_manual_mask'] = True
                # 如果手动绘制了蒙版，则禁用SAM自动分割
                config['sam'] = False
                print(f"使用手动绘制的蒙版: {mask_path}")
        else:
            # 蒙版为空，启用SAM
            config['use_manual_mask'] = False
            config['sam'] = use_sam
            print("手动蒙版为空，将使用SAM智能分割")
    else:
        # 没有手动蒙版，使用SAM设置
        config['use_manual_mask'] = False
        config['sam'] = use_sam
    
    # 更新其他参数
    config['backbone'] = backbone
    config['light_direction'] = light_direction
    config['ambient_strength'] = float(ambient_strength)
    config['diffuse_strength'] = float(diffuse_strength)
    config['sam'] = use_sam if sketch_canvas is None or "mask" not in sketch_canvas else False
    config['num_steps'] = int(num_steps)
    config['controlnet_scale'] = float(controlnet_scale)
    config['prompt'] = prompt
    config['seed'] = int(seed)
    config['top_k'] = int(top_k)
    config['featureweights1'] = float(feature_weight1)
    config['featureweights2'] = float(feature_weight2)
    config['featureweights3'] = float(feature_weight3)
    config['scale'] = float(scale)
    
    # 生成唯一的输出文件名
    timestamp = int(time.time())
    output_file = f"output_{timestamp}.png"
    config['output_file'] = output_file
    
    # 保存更新后的配置
    save_config(config)
    
    return config

# 处理生成请求
def generate(obj, texture, backbone, light_direction, ambient_strength, diffuse_strength, 
             use_sam, num_steps, controlnet_scale, prompt, seed, top_k,
             feature_weight1, feature_weight2, feature_weight3,
             custom_obj=None, custom_texture=None, sketch_canvas=None, scale=None):
    
    # 更新配置
    config = update_config(obj, texture, backbone, light_direction, ambient_strength, diffuse_strength,
                 use_sam, num_steps, controlnet_scale, prompt, seed, top_k,
                 feature_weight1, feature_weight2, feature_weight3,
                 custom_obj, custom_texture, sketch_canvas, scale)
    
    # 运行SynergyFrame
    try:
        print("正在启动SynergyFrame...")
        output_path, preview_path = run_synergyframe()
        print("处理完成！")
        
        # 返回结果图像
        if os.path.exists(output_path) and os.path.exists(preview_path):
            return Image.open(output_path), Image.open(preview_path)
        else:
            return None, None
    except Exception as e:
        print(f"生成过程中出错: {str(e)}")
        return None, None

# 预览蒙版
def preview_mask(sketch_canvas):
    if sketch_canvas is None or "mask" not in sketch_canvas:
        return None
    
    try:
        # 处理蒙版并获取预览
        _, preview_image = process_manual_mask(sketch_canvas)
        return preview_image
    except Exception as e:
        print(f"预览蒙版时出错: {str(e)}")
        return None

# 加载初始配置
initial_config = load_config()

# 创建Gradio界面
with gr.Blocks(title="SynergyFrame演示", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # SynergyFrame: 局部风格化统一框架演示
    
    该演示展示了SynergyFrame的能力 - 一个能够生成**高保真、物理真实感**纹理的统一框架。
    通过协同多种控制信号（几何、光照、风格），实现无缝、精细的局部风格化。
    """)
    
    # 存储当前选择的图像和蒙版
    current_image = gr.State(None)
    current_mask = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 输入选择")
            
            with gr.Tabs() as input_tabs:
                with gr.Tab("从预设选择", id="preset_tab") as preset_tab:
                    obj_dropdown = gr.Dropdown(
                        choices=get_object_options(),
                        value=initial_config.get('obj', '') + ".jpg" if initial_config.get('obj') else None,
                        label="目标物体"
                    )
                    texture_dropdown = gr.Dropdown(
                        choices=get_texture_options(),
                        value=initial_config.get('texture', '') + ".jpg" if initial_config.get('texture') else None,
                        label="风格纹理"
                    )
                
                with gr.Tab("上传自定义图像", id="upload_tab") as upload_tab:
                    custom_obj_upload = gr.File(label="上传自定义物体图像")
                    custom_texture_upload = gr.File(label="上传自定义纹理图像")
            
            backbone_radio = gr.Radio(
                choices=["Inpaint", "Img2Img"],
                value=initial_config.get('backbone', 'Inpaint'),
                label="生成模式"
            )
            
            with gr.Accordion("光照设置", open=False):
                light_direction = gr.Dropdown(
                    choices=["top", "top_left", "top_right", "left", "right", "front", "front_top", "dramatic"],
                    value=initial_config.get('light_direction', 'dramatic'),
                    label="光照方向"
                )
                ambient_strength = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.1,
                    value=initial_config.get('ambient_strength', 0.8),
                    label="环境光强度"
                )
                diffuse_strength = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.1,
                    value=initial_config.get('diffuse_strength', 0.4),
                    label="漫反射强度"
                )
            
            with gr.Accordion("高级设置", open=False):
                use_sam = gr.Checkbox(
                    value=initial_config.get('sam', False),
                    label="使用SAM进行分割"
                )
                num_steps = gr.Slider(
                    minimum=20, maximum=100, step=1,
                    value=initial_config.get('num_steps', 50),
                    label="推理步数"
                )
                controlnet_scale = gr.Slider(
                    minimum=0.1, maximum=1.5, step=0.1,
                    value=initial_config.get('controlnet_scale', 0.9),
                    label="ControlNet权重"
                )
                prompt = gr.Textbox(
                    value=initial_config.get('prompt', ''),
                    label="提示词",
                    placeholder="可选：添加附加提示词引导生成"
                )
                seed = gr.Number(
                    value=initial_config.get('seed', 42),
                    label="随机种子",
                    precision=0
                )
                top_k = gr.Slider(
                    minimum=1, maximum=20, step=1,
                    value=initial_config.get('top_k', 8),
                    label="Top-K注意力"
                )
                scale = gr.Slider(
                    minimum=0.1, maximum=2.0, step=0.1,
                    value=initial_config.get('scale', 1.0),
                    label="Scale"
                )
                
                with gr.Row():
                    feature_weight1 = gr.Slider(
                        minimum=0.0, maximum=3.0, step=0.1,
                        value=initial_config.get('featureweights1', 1.0),
                        label="边界权重"
                    )
                    feature_weight2 = gr.Slider(
                        minimum=0.0, maximum=3.0, step=0.1,
                        value=initial_config.get('featureweights2', 1.0),
                        label="梯度权重"
                    )
                    feature_weight3 = gr.Slider(
                        minimum=0.0, maximum=3.0, step=0.1,
                        value=initial_config.get('featureweights3', 1.0),
                        label="深度权重"
                    )
            
            generate_btn = gr.Button("生成", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 蒙版绘制与预览")
            
            # 添加蒙版绘制区域
            with gr.Tabs() as mask_tabs:
                with gr.Tab("手动绘制蒙版"):
                    # 显示当前图像并允许绘制
                    sketch_canvas = gr.ImageMask(
                        label="绘制蒙版区域",
                        brush_radius=20,
                        height=500,
                        interactive=True
                    )
                    
                    # 加载图像按钮
                    with gr.Row():
                        load_preset_btn = gr.Button("加载预设图像")
                        load_custom_btn = gr.Button("加载上传图像")
                        clear_sketch_btn = gr.Button("清除绘制")
                    
                    # 蒙版预览
                    mask_preview = gr.Image(
                        label="蒙版预览",
                        height=280
                    )
                    
                    # 添加预览按钮
                    preview_mask_btn = gr.Button("预览蒙版效果")
                
                with gr.Tab("SAM智能分割"):
                    # 使用SAM的点击式分割
                    sam_image = gr.Image(
                        label="点击前景/背景点",
                        tool="sketch", 
                        height=750,
                        interactive=True
                    )
                    
                    with gr.Row():
                        sam_load_preset_btn = gr.Button("加载预设图像")
                        sam_load_custom_btn = gr.Button("加载上传图像")
                        sam_clear_btn = gr.Button("清除点")
                    
                    with gr.Row():
                        sam_foreground_btn = gr.Button("设为前景点", variant="primary")
                        sam_background_btn = gr.Button("设为背景点", variant="secondary")
                        sam_generate_btn = gr.Button("生成SAM蒙版")
            
            gr.Markdown("### 结果")
            result_image = gr.Image(label="生成结果", height=750)
            preview_grid = gr.Image(label="输入预览", height=180)
    
    # 设置标签切换事件处理
    preset_tab.select(
        fn=on_preset_tab_select,
        inputs=[],
        outputs=[custom_obj_upload, custom_texture_upload]
    )
    
    upload_tab.select(
        fn=on_upload_tab_select,
        inputs=[],
        outputs=[obj_dropdown, texture_dropdown]
    )
    
    # 加载图像到绘图区域
    load_preset_btn.click(
        fn=on_image_change,
        inputs=[obj_dropdown],
        outputs=[sketch_canvas]
    )
    
    load_custom_btn.click(
        fn=lambda x: on_image_change(x, False),
        inputs=[custom_obj_upload],
        outputs=[sketch_canvas]
    )
    
    clear_sketch_btn.click(
        fn=clear_sketch_and_use_sam,
        inputs=[],
        outputs=[sketch_canvas, use_sam]
    )
    
    # 预览蒙版
    preview_mask_btn.click(
        fn=preview_mask,
        inputs=[sketch_canvas],
        outputs=[mask_preview]
    )
    
    # SAM相关事件处理
    sam_load_preset_btn.click(
        fn=on_image_change,
        inputs=[obj_dropdown],
        outputs=[sam_image]
    )
    
    sam_load_custom_btn.click(
        fn=lambda x: on_image_change(x, False),
        inputs=[custom_obj_upload],
        outputs=[sam_image]
    )
    
    sam_clear_btn.click(
        fn=lambda: None,
        inputs=[],
        outputs=[sam_image]
    )
    
    # 设置生成事件处理
    generate_btn.click(
        fn=generate,
        inputs=[
            obj_dropdown, texture_dropdown, backbone_radio,
            light_direction, ambient_strength, diffuse_strength,
            use_sam, num_steps, controlnet_scale, prompt, seed, top_k,
            feature_weight1, feature_weight2, feature_weight3,
            custom_obj_upload, custom_texture_upload, sketch_canvas, scale
        ],
        outputs=[result_image, preview_grid]
    )

# 启动应用
if __name__ == "__main__":
    # 简单启动，不使用队列和进度跟踪
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860) 