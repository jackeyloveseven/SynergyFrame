#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import subprocess
from tqdm import tqdm

def load_config(config_path='config.json'):
    """从JSON配置文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"已从 {config_path} 加载配置")
        return config
    except Exception as e:
        print(f"加载配置文件 {config_path} 时出错: {str(e)}")
        return {}

# 从config.json加载配置
config = load_config()

# 配置参数
input_dir = config.get('input_dir', 'demo_assets/input_imgs')
texture_dir = config.get('texture_dir', 'demo_assets/texture')
output_dir =  'copy/'
model = config.get('model','SynergyFrame.py')
# 确保路径末尾有斜杠
if not input_dir.endswith('/'):
    input_dir += '/'
if not texture_dir.endswith('/'):
    texture_dir += '/'
if not output_dir.endswith('/'):
    output_dir += '/'

print(f"使用输入目录: {input_dir}")
print(f"使用纹理目录: {texture_dir}")
print(f"使用输出目录: {output_dir}")

# 获取输入目录中的物体文件名（不含扩展名）
objs = []
for file in os.listdir(input_dir):
    if os.path.isfile(os.path.join(input_dir, file)):
        # 获取文件名前缀（假设文件名格式为"前缀_其他内容.扩展名"）
        prefix = file.split('_')[0]
        if prefix not in objs:
            objs.append(prefix)

# 获取纹理目录中的纹理文件名（不含扩展名）
textures = []
for file in os.listdir(texture_dir):
    if os.path.isfile(os.path.join(texture_dir, file)):
        # 获取文件名前缀（假设文件名格式为"前缀_其他内容.扩展名"）
        prefix = file.split('_')[0]
        if prefix not in textures:
            textures.append(prefix)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 打印批处理信息
print(f"找到 {len(objs)} 个物体和 {len(textures)} 个材质，共 {len(objs) * len(textures)} 个组合")

# 创建进度条
total_combinations = len(objs) * len(textures)
progress_bar = tqdm(total=total_combinations, desc="处理进度")

# 批量处理
for texture in textures:
    obj = '18.png'
    # 构建输出文件名
    output_file = os.path.join(output_dir, f"obj_{obj}_texture_{texture}.png")
    
    # 构建命令 - 只传递obj、texture和output_file参数
    cmd = [
        "python", model,
        "--obj", obj,
        "--texture", texture,
        "--output_file", output_file
    ]
    
    # 执行命令
    print(f"\n处理: 物体 {obj} + 材质 {texture}")
    try:
        subprocess.run(cmd, check=True)
        print(f"已保存: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"处理失败: {e}")
    
    # 更新进度条
    progress_bar.update(1)

# 关闭进度条
progress_bar.close()

print(f"\n批处理完成！所有输出已保存到 {output_dir}")