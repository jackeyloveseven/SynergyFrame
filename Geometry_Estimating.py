import numpy as np
import cv2
from scipy import ndimage
from skimage import feature, filters, morphology

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
    
    def _extract_multi_scale_edges(self, image):
        """
        多尺度边缘检测 - 专门用于捕捉depth模型难以检测的细节
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        gray = gray.astype(np.float32) / 255.0
        
        # 多尺度边缘检测，专注于细节增强
        edge_maps = []
        scales = [0.8, 1.5, 3.0, 6.0]  # 更细致的尺度划分
        
        for scale in scales:
            # 高斯平滑
            sigma = scale * 0.6
            smoothed = cv2.GaussianBlur(gray, (0, 0), sigma)
            
            # 增强的Canny边缘检测，使用更低的阈值捕捉更多细节
            edges_canny = feature.canny(smoothed, sigma=sigma, low_threshold=0.05, high_threshold=0.15)
            
            # Sobel边缘检测
            grad_x = cv2.filter2D(smoothed, -1, self.sobel_x)
            grad_y = cv2.filter2D(smoothed, -1, self.sobel_y)
            edges_sobel = np.sqrt(grad_x**2 + grad_y**2)
            
            # 拉普拉斯边缘检测，对细节敏感
            edges_laplacian = np.abs(cv2.filter2D(smoothed, -1, self.laplacian))
            
            # 三种边缘检测的加权融合
            combined_edges = (edges_canny.astype(np.float32) * 0.4 + 
                            edges_sobel * 0.4 + 
                            edges_laplacian * 0.2)
            edge_maps.append(combined_edges)
        
        # 融合多尺度边缘，给细尺度更高权重
        weights = np.array([0.4, 0.35, 0.2, 0.05])
        multi_scale_edges = np.zeros_like(edge_maps[0])
        
        for i, edge_map in enumerate(edge_maps):
            multi_scale_edges += weights[i] * edge_map
            
        return multi_scale_edges
    
    def _compute_structure_tensor_features(self, image):
        """
        计算结构张量特征 - 识别局部几何特征强度
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        gray = gray.astype(np.float32) / 255.0
        
        # 计算梯度
        grad_x = cv2.filter2D(gray, -1, self.sobel_x)
        grad_y = cv2.filter2D(gray, -1, self.sobel_y)
        
        # 结构张量分量
        Ixx = grad_x * grad_x
        Ixy = grad_x * grad_y
        Iyy = grad_y * grad_y
        
        # 高斯加权
        sigma = 1.5
        Ixx = cv2.GaussianBlur(Ixx, (0, 0), sigma)
        Ixy = cv2.GaussianBlur(Ixy, (0, 0), sigma)
        Iyy = cv2.GaussianBlur(Iyy, (0, 0), sigma)
        
        # 计算特征值
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        
        # 避免除零
        trace = np.maximum(trace, 1e-10)
        
        # 计算相干性和各向异性
        coherence = (trace - 2 * np.sqrt(np.maximum(trace**2 - 4 * det, 0))) / trace
        anisotropy = np.sqrt(np.maximum(trace**2 - 4 * det, 0)) / trace
        
        # 结合相干性和各向异性作为结构强度指标
        structure_strength = coherence * anisotropy
        
        return structure_strength
    
    def _adaptive_contour_enhancement(self, depth_map, reference_image):
        """
        自适应轮廓增强 - 专门针对depth模型的弱点进行增强
        """
        if depth_map.ndim == 3:
            depth = depth_map[:, :, 0].copy()
        else:
            depth = depth_map.copy()
            
        depth = depth.astype(np.float32)
        
        # 提取多尺度边缘特征
        edge_features = self._extract_multi_scale_edges(reference_image)
        
        # 计算结构张量特征
        structure_features = self._compute_structure_tensor_features(reference_image)
        
        # 计算深度梯度，识别当前深度图的弱点
        depth_grad_x = cv2.filter2D(depth, -1, self.sobel_x)
        depth_grad_y = cv2.filter2D(depth, -1, self.sobel_y)
        depth_gradient_mag = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
        
        # 归一化特征
        edge_features = cv2.normalize(edge_features, None, 0, 1, cv2.NORM_MINMAX)
        structure_features = cv2.normalize(structure_features, None, 0, 1, cv2.NORM_MINMAX)
        depth_gradient_mag = cv2.normalize(depth_gradient_mag, None, 0, 1, cv2.NORM_MINMAX)
        
        # 计算增强权重：在图像有强结构但深度梯度弱的地方增强更多
        enhancement_weight = (
            edge_features * 2.0 +  # 边缘区域
            structure_features * 1.5 +  # 结构强的区域
            (1 - depth_gradient_mag) * 1.0  # 深度梯度弱的区域（需要增强的地方）
        )
        
        # 平滑增强权重
        enhancement_weight = cv2.GaussianBlur(enhancement_weight, (5, 5), 1.0)
        
        # 使用拉普拉斯算子检测需要增强的局部变化
        laplacian_response = cv2.filter2D(depth, -1, self.laplacian)
        laplacian_response = np.abs(laplacian_response)
        
        # 自适应阈值：基于图像内容动态调整
        threshold = np.percentile(laplacian_response, 60) * 0.4
        
        # 创建增强掩码
        enhancement_mask = (laplacian_response > threshold).astype(np.float32)
        enhancement_mask = cv2.GaussianBlur(enhancement_mask, (3, 3), 0.5)
        
        # 应用增强 - 增强幅度根据特征权重调整
        enhancement_amount = enhancement_weight * enhancement_mask * 25.0
        enhanced_depth = depth + enhancement_amount
        
        return enhanced_depth

    def enhance(self, depth_map, reference_image):
        """
        通过引用图像的多尺度边缘特征来增强深度图。
        这种方法可以更好地捕捉深度模型本身可能忽略的精细几何细节。

        Args:
            depth_map (np.ndarray): 输入深度图 (H,W,3) or (H,W)
            reference_image (np.ndarray): 参考RGB图像 (H,W,3)

        Returns:
            np.ndarray: 增强后的深度图，为float32数组以保持精度。
        """
        # 确保深度图格式正确
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]
        original_depth = depth_map.astype(np.float32)

        # 1. 对原始深度图进行平滑处理，以减少噪声，同时保留主要结构
        smoothed_depth = cv2.bilateralFilter(original_depth, 9, 50, 50)

        # 2. 从参考图像中提取详细的多尺度边缘特征
        # 这些特征将指导深度增强
        edge_features = self._extract_multi_scale_edges(reference_image)
        
        # 3. 将边缘特征归一化，用作增强图
        enhancement_map = cv2.normalize(edge_features, None, 0, 1, cv2.NORM_MINMAX)

        # 4. 定义增强强度
        # 为保持一致性，使用 feature_weights[2]，但采用更直接的缩放因子。
        enhancement_strength = self.feature_weights[2] * 20.0 # 这个因子可以调整

        # 5. 计算增强量
        # 增强与边缘强度成正比
        enhancement_amount = enhancement_map * enhancement_strength

        # 6. (可选) 应用内容感知蒙版，将增强限制在特定的深度范围内
        content_mask = self._generate_content_aware_mask(original_depth)
        enhancement_amount *= content_mask

        # 7. 将增强量添加到平滑后的深度图中
        enhanced_depth = smoothed_depth + enhancement_amount

        # 8. 最终的一致性检查和裁剪
        # 使用现有的 _preserve_depth_consistency 方法进行稳健的收尾
        final_depth = self._preserve_depth_consistency(enhanced_depth, original_depth)
        
        final_depth = np.clip(final_depth, 0, 255)

        # 打印一些调试信息
        max_enhancement = np.max(enhancement_amount)
        mean_enhancement = np.mean(enhancement_amount[enhancement_amount > 0]) if np.any(enhancement_amount > 0) else 0
        
        print(f"🔧 新的增强方法 - 最大增强: {max_enhancement:.2f}, 平均增强: {mean_enhancement:.2f}")

        return final_depth.astype(np.float32)
    
    def _preserve_depth_consistency(self, enhanced_depth, original_depth):
        """
        保持深度一致性 - 确保增强后的深度图在空间上连贯
        """
        # 双边滤波保持边缘的同时平滑噪声
        enhanced = cv2.bilateralFilter(enhanced_depth.astype(np.float32), 9, 40, 40)
        
        # 限制增强幅度，避免过度增强
        diff = enhanced - original_depth
        max_enhancement = np.std(original_depth) * 0.6  # 限制增强幅度
        diff = np.clip(diff, -max_enhancement, max_enhancement)
        
        consistent_depth = original_depth + diff
        
        # 确保深度值在合理范围内
        consistent_depth = np.clip(consistent_depth, 0, 255)
        
        return consistent_depth

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
                 ambient_strength=0.6,  # 提高环境光，减少阴影对比度
                 diffuse_strength=0.5,  # 降低漫反射强度，避免过强阴影
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
        Apply directional lighting to the image with improved shadow handling.
        
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
        
        # 🔥 修复阴影问题：更好的阴影处理
        # 1. 限制阴影的最暗程度，避免大片灰色阴影
        min_shadow_level = 0.3  # 阴影最暗不超过30%
        
        # 2. 平滑阴影过渡，但保持更多细节
        diffuse_smooth = cv2.GaussianBlur(diffuse, (5, 5), 1.0)
        diffuse = diffuse * 0.7 + diffuse_smooth * 0.3  # 混合原始和平滑的漫反射
        
        # 3. 重新映射diffuse值，确保阴影不会太暗
        diffuse = np.clip(diffuse, -1, 1)  # 确保在有效范围内
        diffuse = (diffuse + 1) * 0.5  # 映射到[0,1]
        diffuse = np.clip(diffuse, min_shadow_level, 1.0)  # 限制最暗程度
        
        diffuse = diffuse[..., np.newaxis]
        
        # 4. 改进的光照模型：减少环境光，增强对比度
        ambient = np.ones_like(diffuse) * self.ambient_strength
        
        # 使用更自然的光照混合
        shading = ambient + self.diffuse_strength * (diffuse - min_shadow_level) / (1 - min_shadow_level)
        shading = np.clip(shading, min_shadow_level, 1.5)  # 允许一定程度的过曝，但限制阴影
        
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