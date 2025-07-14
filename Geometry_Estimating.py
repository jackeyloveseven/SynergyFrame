import numpy as np
import cv2

class MultiScaleDepthEnhancement:
    """
    Multi-Scale Depth Enhancement Module (MSDEM)
    
    This module enhances depth maps by incorporating multi-scale geometric features:
    1. Local Structure Enhancement (LSE): Extracts and enhances local geometric structures
    2. Gradient Structure Integration (GSI): Incorporates depth gradients and surface roughness
    3. Adaptive Feature Fusion (AFF): Combines features with content-aware weighting
    """
    
    def __init__(self, 
                 edge_low_threshold=50,
                 edge_high_threshold=150,
                 edge_gaussian_kernel=(5, 5),
                 edge_gaussian_sigma=1,
                 depth_gaussian_kernel=(11, 11),
                 depth_gaussian_sigma=3,
                 depth_valid_range=(5, 250),
                 featureweights=[1.0, 1.0, 1.0]):
        """
        Initialize the MSDEM module.
        
        Args:
            edge_low_threshold (int): Lower threshold for Canny edge detection
            edge_high_threshold (int): Higher threshold for Canny edge detection
            edge_gaussian_kernel (tuple): Kernel size for edge smoothing
            edge_gaussian_sigma (float): Sigma for edge smoothing
            depth_gaussian_kernel (tuple): Kernel size for depth mask smoothing
            depth_gaussian_sigma (float): Sigma for depth mask smoothing
            depth_valid_range (tuple): Valid depth range for enhancement (min, max)
            feature_weights (tuple): Weights for (edge, gradient, structure) features
        """
        self.edge_params = {
            'low_threshold': edge_low_threshold,
            'high_threshold': edge_high_threshold,
            'gaussian_kernel': edge_gaussian_kernel,
            'gaussian_sigma': edge_gaussian_sigma
        }
        
        self.depth_params = {
            'gaussian_kernel': depth_gaussian_kernel,
            'gaussian_sigma': depth_gaussian_sigma,
            'valid_range': depth_valid_range
        }
        
        self.feature_weights = [i/100 for i in featureweights]
        
        # 定义 Sobel 和 Laplacian 核
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        self.laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    
    def _extract_boundary_features(self, img):
        """Local Structure Enhancement: Extract boundary features using Canny edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 
                         self.edge_params['low_threshold'],
                         self.edge_params['high_threshold'])
        edges = cv2.GaussianBlur(edges, 
                                self.edge_params['gaussian_kernel'],
                                self.edge_params['gaussian_sigma'])
        return edges.astype(np.float32) / 255.0 * 0.007
    
    def _compute_gradient_features(self, depth):
        """Gradient Structure Integration: Compute depth gradients and surface roughness"""
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32)
        
        # 1. 降采样平滑处理
        h, w = depth.shape
        scale_factor = 4  # 降采样比例，更大的值意味着更强的平滑
        small_w, small_h = w//scale_factor, h//scale_factor
        
        # 降采样
        small_depth = cv2.resize(depth, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # 平滑处理
        small_depth = cv2.GaussianBlur(small_depth, (11, 11), 3.0)
        small_depth = cv2.bilateralFilter(small_depth, 9, 25, 25)
        
        # 升采样回原始尺寸
        depth_smooth = cv2.resize(small_depth, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. 使用向量化操作计算大间隔梯度 - 加速处理
        gradient_step = 5  # 使用更大的梯度步长
        
        # 添加填充
        padded_depth = cv2.copyMakeBorder(depth_smooth, gradient_step, gradient_step, 
                                        gradient_step, gradient_step, cv2.BORDER_REFLECT)
        
        # 使用numpy的切片操作代替循环 - 水平梯度
        grad_x = (padded_depth[gradient_step:-gradient_step, 2*gradient_step:] - 
                 padded_depth[gradient_step:-gradient_step, :-2*gradient_step]) / (2*gradient_step)
        
        # 垂直梯度
        grad_y = (padded_depth[2*gradient_step:, gradient_step:-gradient_step] - 
                 padded_depth[:-2*gradient_step, gradient_step:-gradient_step]) / (2*gradient_step)
        
        # 计算梯度幅度
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 计算拉普拉斯算子 - 使用向量化操作
        center = padded_depth[gradient_step:-gradient_step, gradient_step:-gradient_step]
        top = padded_depth[:-2*gradient_step, gradient_step:-gradient_step]
        bottom = padded_depth[2*gradient_step:, gradient_step:-gradient_step]
        left = padded_depth[gradient_step:-gradient_step, :-2*gradient_step]
        right = padded_depth[gradient_step:-gradient_step, 2*gradient_step:]
        
        # 拉普拉斯计算
        roughness = np.abs((top + bottom + left + right) - 4 * center)
        
        # 3. 平滑梯度和粗糙度
        grad_magnitude = cv2.GaussianBlur(grad_magnitude, (5, 5), 1.0)
        roughness = cv2.GaussianBlur(roughness, (5, 5), 1.0)
        
        # 归一化特征
        grad_magnitude_norm = cv2.normalize(grad_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        roughness_norm = cv2.normalize(roughness, None, 0, 1, cv2.NORM_MINMAX)
        
        # 4. 组合特征
        combined_features = (grad_magnitude_norm * 0.7 + roughness_norm * 0.3)
        
        # 5. 最后平滑一次以消除任何剩余的伪影
        combined_features = cv2.bilateralFilter(combined_features, 7, 0.1, 7)
        combined_features = cv2.normalize(combined_features, None, 0, 255, cv2.NORM_MINMAX)
        
        return combined_features.astype(np.float32)
    
    def _generate_content_aware_mask(self, depth):
        """Adaptive Feature Fusion: Generate content-aware mask for feature integration"""
        min_depth, max_depth = self.depth_params['valid_range']
        mask = np.logical_and(depth > min_depth, depth < max_depth).astype(np.float32)
        return cv2.GaussianBlur(mask, 
                               self.depth_params['gaussian_kernel'],
                               self.depth_params['gaussian_sigma'])
    
    def _enhance_local_structure(self, depth):
        """Enhance local geometric structures using gradient information"""
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32)
        
        # 1. 降采样平滑处理
        h, w = depth.shape
        scale_factor = 4  # 降采样比例，更大的值意味着更强的平滑
        small_w, small_h = w//scale_factor, h//scale_factor
        
        # 降采样
        small_depth = cv2.resize(depth, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # 平滑处理
        small_depth = cv2.GaussianBlur(small_depth, (11, 11), 3.0)
        small_depth = cv2.bilateralFilter(small_depth, 9, 25, 25)
        
        # 升采样回原始尺寸
        depth_smooth = cv2.resize(small_depth, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. 使用向量化操作计算大间隔梯度
        gradient_step = 5
        
        # 添加填充
        padded_depth = cv2.copyMakeBorder(depth_smooth, gradient_step, gradient_step, 
                                        gradient_step, gradient_step, cv2.BORDER_REFLECT)
        
        # 使用numpy的切片操作代替循环 - 水平梯度
        grad_x = (padded_depth[gradient_step:-gradient_step, 2*gradient_step:] - 
                 padded_depth[gradient_step:-gradient_step, :-2*gradient_step]) / (2*gradient_step)
        
        # 垂直梯度
        grad_y = (padded_depth[2*gradient_step:, gradient_step:-gradient_step] - 
                 padded_depth[:-2*gradient_step, gradient_step:-gradient_step]) / (2*gradient_step)
        
        # 计算梯度幅度
        grad = np.sqrt(grad_x**2 + grad_y**2)
        
        # 3. 平滑梯度以进一步减少伪影
        grad = cv2.GaussianBlur(grad, (7, 7), 1.5)
        
        # 4. 标准化梯度
        grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
        
        # 5. 应用内容感知掩码 
        content_mask = self._generate_content_aware_mask(depth)
        
        # 6. 增强深度
        enhancement_factor = self.feature_weights[2]
        enhanced = depth + enhancement_factor * grad_norm * content_mask
        
        # 7. 最后应用双边滤波以平滑任何引入的伪影
        enhanced = cv2.bilateralFilter(enhanced, 5, 10, 10)
        
        return np.clip(enhanced, 0, 255).astype(np.float32)
    
    def enhance(self, depth_map, reference_image):
        """
        Enhance depth map using multi-scale feature fusion.
        
        Args:
            depth_map (np.ndarray): Input depth map (H,W,3) or (H,W)
            reference_image (np.ndarray): Reference RGB image (H,W,3)
            
        Returns:
            np.ndarray: Enhanced depth map as a float32 array to preserve precision.
        """
        # Ensure depth map is in correct format
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]
        depth_map = depth_map.astype(np.float32)
        
        # Extract multi-scale features
        boundary_features = self._extract_boundary_features(reference_image)
        gradient_features = self._compute_gradient_features(depth_map)
        
        # Resize features to match depth map
        boundary_features = cv2.resize(boundary_features, 
                                     (depth_map.shape[1], depth_map.shape[0]))
        gradient_features = cv2.resize(gradient_features.astype(np.float32), 
                                    (depth_map.shape[1], depth_map.shape[0]))
        
        # Multi-scale feature fusion
        w1, w2, _ = self.feature_weights
        enhanced_depth = depth_map * (1 - w1 - w2) + \
                        w1 * boundary_features * 255 + \
                        w2 * gradient_features
        
        # Final structure enhancement
        enhanced_depth = self._enhance_local_structure(enhanced_depth)
        
        # The returned depth is now float32, preventing quantization.
        return enhanced_depth

# 使用示例:
"""
msdem = MultiScaleDepthEnhancement(
    edge_low_threshold=50,
    edge_high_threshold=150,
    feature_weights=(0.008, 0.008, 0.6)
)

# depth和raw_image的形状都是(H,W,3)
enhanced_depth = msdem.enhance(depth, raw_image)
"""

class DirectionalShadingModule:
    """
    Directional Shading Module for lighting direction simulation
    
    This module implements Ambient + Diffuse lighting model with controllable light direction:
    1. Surface Normal Estimation from depth map
    2. Ambient + Diffuse lighting (Blinn-Phong inspired)
    3. Directional shading computation
    """
    
    def __init__(self, 
                 ambient_strength=0.4,
                 diffuse_strength=0.7,
                 normal_smooth_kernel=(7, 7),
                 normal_smooth_sigma=2.0,
                 shadow_softness=0.1):
        """
        Initialize the DirectionalShadingModule.
        
        Args:
            ambient_strength (float): Strength of ambient lighting [0,1]
            diffuse_strength (float): Strength of diffuse lighting [0,1]
            normal_smooth_kernel (tuple): Kernel size for normal map smoothing
            normal_smooth_sigma (float): Sigma for normal map smoothing
            shadow_softness (float): Factor to soften shadow transitions
        """
        self.ambient_strength = ambient_strength
        self.diffuse_strength = diffuse_strength
        self.shadow_softness = shadow_softness
        self.normal_params = {
            'smooth_kernel': normal_smooth_kernel,
            'smooth_sigma': normal_smooth_sigma
        }
        
        # 预定义 Sobel 核用于法线估计
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    def estimate_normals(self, depth_map):
        """
        Estimate surface normals from depth map.
        
        Args:
            depth_map (np.ndarray): Input depth map (H,W) or (H,W,1)
            
        Returns:
            np.ndarray: Normal map (H,W,3)
        """
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]
        depth_map = depth_map.astype(np.float32)
        
        # 1. 应用强度的平滑预处理 - 这一步至关重要
        # 降采样+平滑+上采样技术，完全消除伪影
        h, w = depth_map.shape
        # 降采样到更小的尺寸 - 降到1/4
        scale_factor = 4
        small_w, small_h = w//scale_factor, h//scale_factor
        small_depth = cv2.resize(depth_map, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # 强力平滑处理
        small_depth = cv2.GaussianBlur(small_depth, (11, 11), 3.0)
        small_depth = cv2.bilateralFilter(small_depth, 9, 25, 25)
        small_depth = cv2.GaussianBlur(small_depth, (7, 7), 2.0)
        
        # 升采样回原始尺寸，使用Lanczos插值（对边缘有更好的处理）
        smooth_depth = cv2.resize(small_depth, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. 向量化的法线计算
        # 定义采样距离
        sample_dist = 5
        
        # 填充深度图以便进行切向量计算
        padded_depth = cv2.copyMakeBorder(smooth_depth, sample_dist, sample_dist, 
                                        sample_dist, sample_dist, cv2.BORDER_REFLECT)
        
        # 使用numpy的切片操作计算梯度，比循环快几个数量级
        # 获取各方向的深度值
        center = padded_depth[sample_dist:-sample_dist, sample_dist:-sample_dist]
        left = padded_depth[sample_dist:-sample_dist, :-2*sample_dist]
        right = padded_depth[sample_dist:-sample_dist, 2*sample_dist:]
        top = padded_depth[:-2*sample_dist, sample_dist:-sample_dist]
        bottom = padded_depth[2*sample_dist:, sample_dist:-sample_dist]
        
        # 创建坐标网格（为每个像素提供x,y相对坐标）
        y_coords, x_coords = np.mgrid[-sample_dist:sample_dist+1:2*sample_dist, 
                                       -sample_dist:sample_dist+1:2*sample_dist]
        
        # 创建3D点坐标
        pt_left = np.stack([-sample_dist * np.ones_like(center), np.zeros_like(center), left], axis=-1)
        pt_right = np.stack([sample_dist * np.ones_like(center), np.zeros_like(center), right], axis=-1)
        pt_top = np.stack([np.zeros_like(center), -sample_dist * np.ones_like(center), top], axis=-1)
        pt_bottom = np.stack([np.zeros_like(center), sample_dist * np.ones_like(center), bottom], axis=-1)
        
        # 计算切向量
        tangent_x = pt_right - pt_left
        tangent_y = pt_bottom - pt_top
        
        # 计算叉乘（法线向量）- 使用向量化操作
        normals = np.zeros((h, w, 3), dtype=np.float32)
        
        normals[..., 0] = tangent_x[..., 1] * tangent_y[..., 2] - tangent_x[..., 2] * tangent_y[..., 1]
        normals[..., 1] = tangent_x[..., 2] * tangent_y[..., 0] - tangent_x[..., 0] * tangent_y[..., 2]
        normals[..., 2] = tangent_x[..., 0] * tangent_y[..., 1] - tangent_x[..., 1] * tangent_y[..., 0]
        
        # 确保法线朝向观察者
        flip_mask = (normals[..., 2] < 0)
        normals[flip_mask] = -normals[flip_mask]
        
        # 归一化法线 - 修复广播错误
        norms = np.sqrt(np.sum(normals**2, axis=2))
        
        # 找出非零法线的掩码
        valid_mask = (norms > 1e-10)
        
        # 对于每个有效像素进行归一化
        for i in range(3):  # 对x,y,z分量分别处理
            normals[valid_mask, i] = normals[valid_mask, i] / norms[valid_mask]
        
        # 设置无效像素为默认法线
        normals[~valid_mask] = np.array([0, 0, 1], dtype=np.float32)
        
        # 对法线应用双边滤波以保留边缘并平滑伪影
        normals_x = cv2.bilateralFilter(normals[:,:,0], 7, 0.05, 7)
        normals_y = cv2.bilateralFilter(normals[:,:,1], 7, 0.05, 7)
        normals_z = cv2.bilateralFilter(normals[:,:,2], 7, 0.05, 7)
        
        normals = np.stack((normals_x, normals_y, normals_z), axis=-1)
        
        # 最终归一化
        norms = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
        normals = normals / (norms + 1e-10)
        
        return normals
    
    def apply_lighting(self, image, normals, light_dir, mask=None):
        """
        Apply directional lighting to the image.
        
        Args:
            image (np.ndarray): Input RGB image (H,W,3)
            normals (np.ndarray): Normal map (H,W,3)
            light_dir (np.ndarray): Light direction vector [x,y,z]
            mask (np.ndarray, optional): Binary mask for foreground (255 for foreground)
            
        Returns:
            np.ndarray: Shaded image (H,W,3)
        """
        # 归一化光照方向
        light_dir = np.array(light_dir, dtype=np.float32)
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-10)
        
        # 计算漫反射项 (N·L)
        diffuse = np.sum(normals * light_dir, axis=2)
        
        # 软化阴影过渡
        diffuse = (diffuse + self.shadow_softness) / (1 + self.shadow_softness)
        diffuse = np.clip(diffuse, 0, 1)
        diffuse = diffuse[..., np.newaxis]
        
        # 应用环境光和漫反射
        ambient = np.ones_like(diffuse) * self.ambient_strength
        shading = ambient + self.diffuse_strength * diffuse
        
        # 应用光照到图像
        shaded_image = image.astype(np.float32) * shading
        
        # 如果有mask，只在前景区域应用光照效果
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[..., np.newaxis]
            mask = mask.astype(np.float32) / 255.0
            shaded_image = shaded_image * mask + image.astype(np.float32) * (1 - mask)
        
        return np.clip(shaded_image, 0, 255).astype(np.uint8)
    
    def extract_lighting(self, image, depth_map=None):
        """
        Extract lighting information from original image.
        
        Args:
            image (np.ndarray): Input RGB image
            depth_map (np.ndarray, optional): Depth map for normal estimation
            
        Returns:
            np.ndarray: Extracted lighting map
        """
        # 转换为灰度图
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 如果有深度图，使用深度图估计法线
        if depth_map is not None:
            normals = self.estimate_normals(depth_map)
            # 使用法线信息提取光照
            lighting = np.mean(normals * [0, 0, 1], axis=2)
            lighting = (lighting + 1) * 0.5  # 归一化到 [0,1]
        else:
            # 使用图像梯度估计光照
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            lighting = np.sqrt(grad_x**2 + grad_y**2)
            lighting = cv2.normalize(lighting, None, 0, 1, cv2.NORM_MINMAX)
            
        return (lighting * 255).astype(np.uint8)
    
    def simulate_lighting(self, image, depth_map, light_dir, mask=None):
        """
        Simulate lighting with specified direction.
        
        Args:
            image (np.ndarray): Input RGB image
            depth_map (np.ndarray): Input depth map
            light_dir (list/np.ndarray): Light direction vector [x,y,z]
            mask (np.ndarray, optional): Binary mask for foreground
            
        Returns:
            np.ndarray: Image with simulated lighting
        """
        # 估计表面法线
        normals = self.estimate_normals(depth_map)
        
        # 应用光照
        shaded = self.apply_lighting(image, normals, light_dir, mask)
        
        return shaded


# 使用示例:
"""
# 创建模块实例
dsm = DirectionalShadingModule(
    ambient_strength=0.3,
    diffuse_strength=0.7
)

# 模拟新的光照方向
light_dir = [0.5, 0.5, 1.0]  # 左上方光源
shaded_image = dsm.simulate_lighting(image, depth_map, light_dir)

# 或者仅提取原始光照信息
lighting_map = dsm.extract_lighting(image, depth_map)
""" 