import os
import numpy as np
from src.data.data_loader import DataLoader
from src.visualization.image_display import ImageDisplay

class SecondStageProcessor:
    """二阶段模型数据处理器"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.image_display = ImageDisplay()
    
    def nifti_to_png(self, nifti_path, output_dir, slice_axis=0):
        """
        将NIFTI格式数据转换为PNG格式，用于二阶段模型输入
        
        Args:
            nifti_path: NIFTI文件路径
            output_dir: PNG文件输出目录
            slice_axis: 切片轴，0表示深度轴
            
        Returns:
            png_paths: PNG文件路径列表
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载NIFTI文件
        image_data, affine, header = self.data_loader.load_nifti(nifti_path)
        if image_data is None:
            return []
        
        # 获取图像信息
        depth, height, width = image_data.shape
        
        # 生成PNG文件路径列表
        png_paths = []
        
        # 沿指定轴切片并保存为PNG
        for i in range(depth):
            # 获取切片
            slice_data = self.image_display.get_slice(image_data, i, axis=slice_axis)
            # 生成输出文件名
            output_filename = f"slice_{i:04d}.png"
            output_path = os.path.join(output_dir, output_filename)
            # 保存为PNG
            self.image_display.save_slice_as_png(slice_data, output_path)
            # 添加到路径列表
            png_paths.append(output_path)
        
        return png_paths
    
    def png_to_nifti(self, png_dir, output_nifti_path, reference_nifti_path):
        """
        将PNG格式数据转换回NIFTI格式，用于结果保存和评估
        
        Args:
            png_dir: PNG文件目录
            output_nifti_path: 输出NIFTI文件路径
            reference_nifti_path: 参考NIFTI文件路径（用于获取仿射变换和头部信息）
            
        Returns:
            bool: 转换是否成功
        """
        try:
            # 加载参考NIFTI文件，获取仿射变换和头部信息
            _, affine, header = self.data_loader.load_nifti(reference_nifti_path)
            if affine is None or header is None:
                return False
            
            # 获取PNG文件列表并排序
            png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')])
            if not png_files:
                return False
            
            # 加载第一个PNG文件，获取尺寸
            first_png_path = os.path.join(png_dir, png_files[0])
            first_slice = self.image_display.load_png_as_slice(first_png_path)
            height, width = first_slice.shape
            depth = len(png_files)
            
            # 创建3D图像数据
            image_data = np.zeros((depth, height, width), dtype=np.float32)
            
            # 加载所有PNG文件并填充到3D数据中
            for i, png_file in enumerate(png_files):
                png_path = os.path.join(png_dir, png_file)
                slice_data = self.image_display.load_png_as_slice(png_path)
                image_data[i, :, :] = slice_data
            
            # 保存为NIFTI格式
            success = self.data_loader.save_nifti(image_data, affine, header, output_nifti_path)
            
            return success
        except Exception as e:
            print(f"PNG转NIFTI时出错: {e}")
            return False
    
    def process_first_stage_output(self, first_stage_output, output_dir):
        """
        处理一阶段模型输出，准备二阶段模型输入
        
        Args:
            first_stage_output: 一阶段模型输出，可以是NIFTI文件路径或3D数组
            output_dir: 二阶段模型输入目录（PNG格式）
            
        Returns:
            png_paths: PNG文件路径列表
        """
        if isinstance(first_stage_output, str) and first_stage_output.endswith('.nii'):
            # 如果是NIFTI文件路径
            return self.nifti_to_png(first_stage_output, output_dir)
        elif isinstance(first_stage_output, np.ndarray) and first_stage_output.ndim == 3:
            # 如果是3D数组，先保存为临时NIFTI文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as temp_file:
                temp_nifti_path = temp_file.name
            
            # 创建临时NIFTI文件
            # 注意：这里需要仿射变换和头部信息，实际应用中可能需要从原始图像获取
            # 这里使用默认值
            temp_affine = np.eye(4)
            temp_header = None
            self.data_loader.save_nifti(first_stage_output, temp_affine, temp_header, temp_nifti_path)
            
            # 转换为PNG
            png_paths = self.nifti_to_png(temp_nifti_path, output_dir)
            
            # 删除临时文件
            try:
                os.unlink(temp_nifti_path)
            except:
                pass
            
            return png_paths
        else:
            raise ValueError("first_stage_output必须是NIFTI文件路径或3D numpy数组")
    
    def process_second_stage_output(self, second_stage_output, output_nifti_path, reference_nifti_path):
        """
        处理二阶段模型输出，转换回NIFTI格式
        
        Args:
            second_stage_output: 二阶段模型输出，可以是PNG目录或3D数组
            output_nifti_path: 输出NIFTI文件路径
            reference_nifti_path: 参考NIFTI文件路径
            
        Returns:
            bool: 处理是否成功
        """
        if isinstance(second_stage_output, str) and os.path.isdir(second_stage_output):
            # 如果是PNG目录
            return self.png_to_nifti(second_stage_output, output_nifti_path, reference_nifti_path)
        elif isinstance(second_stage_output, np.ndarray) and second_stage_output.ndim == 3:
            # 如果是3D数组，直接保存为NIFTI
            # 加载参考NIFTI文件，获取仿射变换和头部信息
            _, affine, header = self.data_loader.load_nifti(reference_nifti_path)
            if affine is None or header is None:
                return False
            
            # 保存为NIFTI格式
            return self.data_loader.save_nifti(second_stage_output, affine, header, output_nifti_path)
        else:
            raise ValueError("second_stage_output必须是PNG目录或3D numpy数组")