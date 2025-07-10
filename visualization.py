import glob
import os
import json
from html4vision import Col, imagetable

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

# 使用与batch_process.py相同的路径配置
input_dir = config.get('input_dir', 'demo_assets/input_imgs')
texture_dir = config.get('texture_dir', 'demo_assets/texture')
output_dir = config.get('output_dir', 'batch_outputs/')

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

# 获取所有物体和材质文件名（不含扩展名）
textures = []
for file in os.listdir(texture_dir):
    if os.path.isfile(os.path.join(texture_dir, file)):
        prefix = file.split('_')[0]
        if prefix not in textures:
            textures.append(prefix)

objs = []
for file in os.listdir(input_dir):
    if os.path.isfile(os.path.join(input_dir, file)):
        prefix = file.split('_')[0]
        if prefix not in objs:
            objs.append(prefix)

# 生成第一列作为输入图像参考
cols = []
cols.append(Col('img', '', [''] + [os.path.join(input_dir, f"{obj}_{obj}.png") for obj in objs]))

# 为每个材质生成一列结果
for texture in textures:
    # 第一行显示材质图像
    cur_col = [os.path.join(texture_dir, f"{texture}_{texture}.png")]
    
    # 其余行显示生成的结果
    for obj in objs:
        cur_col.append(os.path.join(output_dir, f"obj_{obj}_texture_{texture}.png"))
    
    cols.append(Col('img', texture, cur_col))

# 生成HTML表格
html_file = "visualization_results.html"
imagetable(cols, out_file=html_file)
print(f"可视化结果已保存至: {html_file}")