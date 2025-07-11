# SynergyFrame

SynergyFrame是一个材质与物体融合系统，能够将不同材质应用到物体上，并生成逼真的渲染效果。

## 功能特点

- 自动深度估计：使用DepthAnythingV2模型自动估计输入图像的深度信息
- 背景移除：支持rembg和SAM模型进行背景移除和物体分割
- 光照模拟：基于深度图和物体遮罩模拟真实光照效果
- 材质融合：使用IP-Adapter和ControlNet技术将材质示例应用到目标物体
- 高质量输出：支持高分辨率图像处理和生成

## 安装

### 环境要求

- Python 3.8+
- CUDA支持的GPU (推荐)

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/username/SynergyFrame.git
cd SynergyFrame
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

或者使用setup.py安装：

```bash
pip install -e .
```

3. 下载预训练模型：

需要下载以下预训练模型：
- DepthAnythingV2模型: `checkpoints/depth_anything_v2_vitb.pth`
- IP-Adapter模型: `sdxl_models/ip-adapter_sdxl_vit-h.bin`
- 图像编码器: `models/image_encoder`
- ControlNet模型: `diffusers/controlnet-depth-sdxl-1.0`

## 使用方法

### 命令行使用

```bash
python SynergyFrame.py --obj 5 --texture 5 --light_direction right --sam true
```

参数说明：
- `--obj`: 目标物体图片名称或编号
- `--texture`: 材质图片名称或编号
- `--light_direction`: 光照方向，可选值：top, top_left, top_right, left, right, front, front_top, dramatic
- `--ambient_strength`: 环境光强度，默认0.8
- `--diffuse_strength`: 漫反射强度，默认1.5
- `--sam`: 是否使用SAM模型进行背景移除，默认false
- `--backbone`: 选择使用的模型骨架，可选值：Img2Img, Inpaint

### 配置文件使用

也可以通过配置文件设置参数：

```bash
python SynergyFrame.py --config config.json
```

配置文件示例 (config.json):
```json
{
  "obj": "5",
  "texture": "5",
  "input_dir": "demo_assets/input_imgs/",
  "texture_dir": "demo_assets/material_exemplars/",
  "depth_dir": "demo_assets/depths",
  "output_file": "synergy_output.png",
  "light_direction": "right",
  "ambient_strength": 0.8,
  "diffuse_strength": 1.5,
  "use_cuda": true,
  "use_mixed_precision": false,
  "use_fp16": false,
  "use_xformers": true,
  "sam": false,
  "backbone": "Img2Img"
}
```

## 项目结构

- `SynergyFrame.py`: 主程序
- `ip_adapter/`: IP-Adapter相关模块
- `depth_anything_v2/`: 深度估计模型
- `Geometry_Estimating.py`: 几何估计和光照模拟模块
- `demo_assets/`: 示例资源
  - `input_imgs/`: 输入图像
  - `material_exemplars/`: 材质示例
  - `depths/`: 深度图输出目录

## 示例

输入物体图像：
![输入物体](demo_assets/input_imgs/5.png)

材质示例：
![材质示例](demo_assets/material_exemplars/5.png)

输出结果：
![输出结果](synergy_output.png)

## 许可证

MIT