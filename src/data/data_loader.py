import nibabel as nib
import numpy as np

class DataLoader:
    """医学图像数据加载器"""
    
    def load_nifti(self, file_path):
        """
        加载NIFTI格式的3D医学图像
        
        Args:
            file_path: NIFTI文件路径
            
        Returns:
            image_data: 加载的图像数据，形状为(深度, 高度, 宽度)
            affine: 图像的仿射变换矩阵
            header: 图像的头部信息
        """
        try:
            # 加载NIFTI文件
            img = nib.load(file_path)
            # 获取图像数据
            image_data = img.get_fdata()
            # 转换维度顺序为(深度, 高度, 宽度)
            image_data = np.transpose(image_data, (2, 1, 0))
            # 获取仿射变换矩阵
            affine = img.affine
            # 获取头部信息
            header = img.header
            
            return image_data, affine, header
        except Exception as e:
            print(f"加载NIFTI文件时出错: {e}")
            return None, None, None
    
    def save_nifti(self, image_data, affine, header, file_path):
        """
        保存数据为NIFTI格式
        
        Args:
            image_data: 要保存的图像数据
            affine: 仿射变换矩阵
            header: 头部信息
            file_path: 保存路径
        """
        try:
            # 转换维度顺序回(x, y, z)
            image_data = np.transpose(image_data, (2, 1, 0))
            # 创建NIFTI图像对象
            img = nib.Nifti1Image(image_data, affine, header)
            # 保存文件
            nib.save(img, file_path)
            return True
        except Exception as e:
            print(f"保存NIFTI文件时出错: {e}")
            return False
    
    def get_image_info(self, header):
        """
        获取图像信息
        
        Args:
            header: 图像头部信息
            
        Returns:
            info: 包含图像信息的字典
        """
        if header is None:
            return {}
        
        info = {
            'dimensions': header.get('dim')[1:4],
            'voxel_size': header.get('pixdim')[1:4],
            'data_type': header.get('datatype'),
            'bit_depth': header.get('bitpix')
        }
        
        return info