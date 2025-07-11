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
        
        print(featureweights)
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
        
        # 计算深度梯度
        grad_x = cv2.filter2D(depth, cv2.CV_32F, self.sobel_x)
        grad_y = cv2.filter2D(depth, cv2.CV_32F, self.sobel_y)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 计算表面粗糙度 (Laplacian)
        roughness = cv2.filter2D(depth, cv2.CV_32F, self.laplacian)
        roughness = np.abs(roughness)
        
        # 组合特征
        combined_features = (grad_magnitude * 0.7 + roughness * 0.3)
        combined_features = cv2.normalize(combined_features, None, 0, 255, cv2.NORM_MINMAX)
        
        # CRITICAL FIX: Return as float32 to prevent quantization artifacts
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
        
        # Extract structural gradients
        sobelx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(sobelx**2 + sobely**2)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
        
        # Content-aware enhancement
        mask = self._generate_content_aware_mask(depth)
        enhanced = depth + self.feature_weights[2] * grad * mask
        
        # CRITICAL FIX: Return as float32 to preserve precision
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
        
        # 使用高斯模糊和平滑深度图以减少噪声
        depth_smooth = cv2.GaussianBlur(depth_map, 
                                      self.normal_params['smooth_kernel'],
                                      self.normal_params['smooth_sigma'])
        
        # 使用双边滤波进一步平滑，同时保留边缘
        depth_smooth = cv2.bilateralFilter(depth_smooth.astype(np.float32), 9, 75, 75)

        # 计算深度梯度
        grad_x = cv2.filter2D(depth_smooth, cv2.CV_32F, self.sobel_x)
        grad_y = cv2.filter2D(depth_smooth, cv2.CV_32F, self.sobel_y)
        
        # 构建法线图 (-grad_x, -grad_y, 1)
        normals = np.dstack([-grad_x, -grad_y, np.ones_like(depth_map)])
        
        # 归一化法线向量
        norm = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
        normals = normals / (norm + 1e-10)
        
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