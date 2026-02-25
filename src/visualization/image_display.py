import numpy as np
import cv2

class ImageDisplay:
    """医学图像显示工具"""
    
    def normalize_slice(self, slice_data, min_val=None, max_val=None):
        """
        归一化图像切片到0-255范围
        
        Args:
            slice_data: 二维图像切片数据
            min_val: 最小值，默认为None（使用数据最小值）
            max_val: 最大值，默认为None（使用数据最大值）
            
        Returns:
            normalized_slice: 归一化后的图像切片
        """
        if min_val is None:
            min_val = np.min(slice_data)
        if max_val is None:
            max_val = np.max(slice_data)
        
        # 归一化到0-1范围
        normalized_slice = (slice_data - min_val) / (max_val - min_val + 1e-8)
        # 转换到0-255范围
        normalized_slice = (normalized_slice * 255).astype(np.uint8)
        
        return normalized_slice
    
    def get_slice(self, image_data, slice_index, axis=0):
        """
        从3D图像中获取指定轴和索引的切片
        
        Args:
            image_data: 3D图像数据，形状为(深度, 高度, 宽度)
            slice_index: 切片索引
            axis: 切片轴，0表示深度轴，1表示高度轴，2表示宽度轴
            
        Returns:
            slice_data: 二维图像切片
        """
        if axis == 0:
            # 沿深度轴切片
            return image_data[slice_index, :, :]
        elif axis == 1:
            # 沿高度轴切片
            return image_data[:, slice_index, :]
        elif axis == 2:
            # 沿宽度轴切片
            return image_data[:, :, slice_index]
        else:
            raise ValueError("axis必须为0、1或2")
    
    def overlay_mask(self, image_slice, mask_slice, alpha=0.5, color=(0, 255, 0)):
        """
        在图像切片上叠加分割掩码
        
        Args:
            image_slice: 原始图像切片
            mask_slice: 分割掩码切片
            alpha: 掩码透明度，0-1之间
            color: 掩码颜色，BGR格式
            
        Returns:
            overlayed_image: 叠加掩码后的图像
        """
        # 确保图像是3通道的
        if len(image_slice.shape) == 2:
            image_slice = cv2.cvtColor(image_slice, cv2.COLOR_GRAY2BGR)
        
        # 创建掩码的彩色版本
        mask_color = np.zeros_like(image_slice)
        mask_color[mask_slice > 0] = color
        
        # 叠加掩码
        overlayed_image = cv2.addWeighted(image_slice, 1 - alpha, mask_color, alpha, 0)
        
        return overlayed_image
    
    def save_slice_as_png(self, slice_data, output_path):
        """
        将图像切片保存为PNG文件
        
        Args:
            slice_data: 二维图像切片数据
            output_path: 输出PNG文件路径
            
        Returns:
            success: 保存是否成功
        """
        # 归一化图像
        normalized_slice = self.normalize_slice(slice_data)
        # 保存为PNG
        return cv2.imwrite(output_path, normalized_slice)
    
    def load_png_as_slice(self, input_path):
        """
        加载PNG文件作为图像切片
        
        Args:
            input_path: 输入PNG文件路径
            
        Returns:
            slice_data: 二维图像切片数据
        """
        # 加载PNG文件
        slice_data = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        return slice_data