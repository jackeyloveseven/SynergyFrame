#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
import random

def copy_images(src_dir, copies=20):
    """
    将源目录中的图片复制指定的次数
    
    参数:
        src_dir: 源图片目录
        copies: 每张图片需要复制的次数
    """
    # 确保源目录存在
    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"错误: 源目录 {src_dir} 不存在!")
        return
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = []
    
    for file in src_path.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    total_images = len(image_files)
    if total_images == 0:
        print(f"错误: 在 {src_dir} 中没有找到图片文件!")
        return
    
    print(f"在 {src_dir} 中找到 {total_images} 张图片")
    
    # 开始复制
    copied_count = 0
    for i in range(1, copies + 1):
        for img_file in image_files:
            # 生成新文件名 (原文件名_copy序号.扩展名)
            new_name = f"{img_file.stem}_copy{i}{img_file.suffix}"
            new_path = src_path / new_name
            
            # 复制文件
            shutil.copy2(img_file, new_path)
            copied_count += 1
            
            # 显示进度
            if copied_count % 100 == 0:
                print(f"已复制 {copied_count} 张图片...")
    
    # 统计结果
    final_count = len(list(src_path.iterdir()))
    print(f"复制完成! 原有图片: {total_images} 张")
    print(f"复制后总计: {final_count} 张")

if __name__ == "__main__":
    # 源图片目录
    src_directory = "/home/lcy/Lucky/samples/src_image"
    
    # 每张图片复制20次
    copy_images(src_directory, 19) 