import numpy as np
import cv2
from scipy.ndimage import zoom

class Preprocessor:
    """医学图像预处理器"""
    
    def normalize_intensity(self, image_data, method='z-score'):
        """
        对医学图像进行亮度归一化
        
        Args:
            image_data: 3D医学图像数据
            method: 归一化方法，可选值：'z-score'（z-score归一化）或'histogram'（直方图均衡化）
            
        Returns:
            normalized_data: 归一化后的图像数据
        """
        if method == 'z-score':
            # z-score归一化
            mean = np.mean(image_data)
            std = np.std(image_data)
            normalized_data = (image_data - mean) / (std + 1e-8)
        elif method == 'histogram':
            # 直方图均衡化（对每个切片单独处理）
            normalized_data = np.zeros_like(image_data)
            depth, height, width = image_data.shape
            for i in range(depth):
                # 获取当前切片
                slice_data = image_data[i, :, :]
                # 归一化到0-255范围
                min_val = np.min(slice_data)
                max_val = np.max(slice_data)
                normalized_slice = (slice_data - min_val) / (max_val - min_val + 1e-8)
                normalized_slice = (normalized_slice * 255).astype(np.uint8)
                # 应用直方图均衡化
                equalized_slice = cv2.equalizeHist(normalized_slice)
                # 转换回原始范围
                normalized_data[i, :, :] = (equalized_slice / 255.0) * (max_val - min_val) + min_val
        else:
            raise ValueError("method必须为'z-score'或'histogram'")
        
        return normalized_data
    
    def resample(self, image_data, original_voxel_size, target_voxel_size):
        """
        统一医学图像的空间分辨率
        
        Args:
            image_data: 3D医学图像数据
            original_voxel_size: 原始体素大小，形状为(深度, 高度, 宽度)
            target_voxel_size: 目标体素大小，形状为(深度, 高度, 宽度)
            
        Returns:
            resampled_data: 重采样后的图像数据
        """
        # 计算缩放因子
        zoom_factors = [orig / target for orig, target in zip(original_voxel_size, target_voxel_size)]
        
        # 使用双线性插值进行重采样
        resampled_data = zoom(image_data, zoom_factors, order=1, mode='nearest')
        
        return resampled_data
    
    def slice_3d_to_2d(self, image_data, axis=0):
        """
        将3D医学图像切分为2D切片
        
        Args:
            image_data: 3D医学图像数据，形状为(深度, 高度, 宽度)
            axis: 切片轴，0表示深度轴，1表示高度轴，2表示宽度轴
            
        Returns:
            slices: 2D切片列表
        """
        slices = []
        
        if axis == 0:
            # 沿深度轴切片
            depth = image_data.shape[0]
            for i in range(depth):
                slices.append(image_data[i, :, :])
        elif axis == 1:
            # 沿高度轴切片
            height = image_data.shape[1]
            for i in range(height):
                slices.append(image_data[:, i, :])
        elif axis == 2:
            # 沿宽度轴切片
            width = image_data.shape[2]
            for i in range(width):
                slices.append(image_data[:, :, i])
        else:
            raise ValueError("axis必须为0、1或2")
        
        return slices
    
    def preprocess_pipeline(self, image_data, original_voxel_size, target_voxel_size=(1, 1, 1), 
                           normalize_method='z-score', slice_axis=0):
        """
        完整的预处理流程
        
        Args:
            image_data: 3D医学图像数据
            original_voxel_size: 原始体素大小
            target_voxel_size: 目标体素大小
            normalize_method: 归一化方法
            slice_axis: 切片轴
            
        Returns:
            slices: 预处理后的2D切片列表
        """
        # 1. 亮度归一化
        normalized_data = self.normalize_intensity(image_data, method=normalize_method)
        
        # 2. 空间分辨率统一
        resampled_data = self.resample(normalized_data, original_voxel_size, target_voxel_size)
        
        # 3. 二维切分
        slices = self.slice_3d_to_2d(resampled_data, axis=slice_axis)
        
        return slices