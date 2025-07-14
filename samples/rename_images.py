#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path

def rename_target_images(src_dir, tgt_dir):
    """
    按照源目录中的文件名称排序，将这些名称赋给目标目录中的文件
    
    参数:
        src_dir: 源图片目录
        tgt_dir: 目标图片目录
    """
    # 确保源目录和目标目录都存在
    src_path = Path(src_dir)
    tgt_path = Path(tgt_dir)
    
    if not src_path.exists():
        print(f"错误: 源目录 {src_dir} 不存在!")
        return
    
    if not tgt_path.exists():
        print(f"错误: 目标目录 {tgt_dir} 不存在!")
        return
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    # 获取源目录中的图片文件并排序
    src_images = []
    for file in src_path.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            src_images.append(file)
    
    src_images.sort()  # 按文件名排序
    
    # 获取目标目录中的图片文件
    tgt_images = []
    for file in tgt_path.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            tgt_images.append(file)
    
    # 确保目标目录中有足够的图片
    if len(tgt_images) < len(src_images):
        print(f"警告: 目标目录中的图片数量({len(tgt_images)})少于源目录({len(src_images)})!")
        print("将只处理目标目录中可用的图片数量")
    
    # 对目标目录中的图片进行排序（可选，取决于是否需要特定顺序）
    tgt_images.sort()
    
    # 开始重命名
    renamed_count = 0
    for i, src_img in enumerate(src_images):
        if i >= len(tgt_images):
            break
            
        # 获取源文件的名称和扩展名
        src_name = src_img.stem
        src_ext = src_img.suffix
        
        # 创建新的文件名 (使用源文件的扩展名)
        new_name = f"{src_name}{src_ext}"
        new_path = tgt_path / new_name
        
        # 如果目标文件已存在，添加后缀以避免覆盖
        counter = 1
        while new_path.exists():
            new_name = f"{src_name}_{counter}{src_ext}"
            new_path = tgt_path / new_name
            counter += 1
        
        # 重命名文件
        old_path = tgt_images[i]
        old_path.rename(new_path)
        renamed_count += 1
        
        # 显示进度
        if renamed_count % 50 == 0:
            print(f"已重命名 {renamed_count} 张图片...")
    
    print(f"重命名完成! 共重命名了 {renamed_count} 张图片")

if __name__ == "__main__":
    # 源图片目录和目标图片目录
    src_directory = "/home/lcy/Lucky/samples/src_image"
    tgt_directory = "/home/lcy/Lucky/samples/tgt_image"
    
    # 执行重命名操作
    rename_target_images(src_directory, tgt_directory) 