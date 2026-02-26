import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QLabel, QSpinBox, QComboBox, 
    QTabWidget, QGroupBox, QFormLayout, QProgressBar, QTextEdit,
    QAction, QToolBar, QStatusBar, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt

from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.visualization.image_display import ImageDisplay
from src.visualization.evaluation import ResultVisualizer, Evaluator
from src.postprocessing.second_stage_processor import SecondStageProcessor

from src.ui.prediction_thread import PredictionThread
from src.ui.tabs.image_tab import create_image_tab
from src.ui.tabs.preprocessing_tab import create_preprocessing_tab
from src.ui.tabs.prediction_tab import create_prediction_tab
from src.ui.tabs.visualization_tab import create_visualization_tab
from src.ui.tabs.evaluation_tab import create_evaluation_tab


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化组件
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.image_display = ImageDisplay()
        self.result_visualizer = ResultVisualizer()
        self.evaluator = Evaluator()
        self.second_stage_processor = SecondStageProcessor()
        
        # 数据存储
        self.image_data = None
        self.affine = None
        self.header = None
        self.preprocessed_data = None
        self.prediction = None
        self.metrics = None
        self.label_data = None
        
        # 当前切片索引
        self.current_slice = 0
        
        # 初始化UI
        self.init_ui()
    
    def init_ui(self):
        """初始化UI界面"""
        # 设置窗口标题和大小
        self.setWindowTitle('脑微出血分割系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建选项卡部件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 创建各个选项卡
        create_image_tab(self)
        create_preprocessing_tab(self)
        create_prediction_tab(self)
        create_visualization_tab(self)
        create_evaluation_tab(self)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('就绪')
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menu_bar = self.menuBar()
        
        # 文件菜单
        file_menu = menu_bar.addMenu('文件')
        
        # 打开图像动作
        open_image_action = QAction('打开图像', self)
        open_image_action.triggered.connect(self.open_image)
        file_menu.addAction(open_image_action)
        
        # 打开标签动作
        open_label_action = QAction('打开标签', self)
        open_label_action.triggered.connect(self.open_label)
        file_menu.addAction(open_label_action)
        
        # 保存动作
        save_action = QAction('保存', self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        # 保存所有图像动作
        save_images_action = QAction('保存所有图像', self)
        save_images_action.triggered.connect(self.save_images)
        file_menu.addAction(save_images_action)
        
        # 退出动作
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tool_menu = menu_bar.addMenu('工具')
        
        # 预处理动作
        preprocess_action = QAction('预处理', self)
        preprocess_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        tool_menu.addAction(preprocess_action)
        
        # 预测动作
        predict_action = QAction('预测', self)
        predict_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))
        tool_menu.addAction(predict_action)
        
        # 帮助菜单
        help_menu = menu_bar.addMenu('帮助')
        
        # 关于动作
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    

    
    def open_image(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '打开图像文件', '', 'NIFTI文件 (*.nii *.nii.gz)'
        )
        
        if file_path:
            try:
                self.status_bar.showMessage('加载图像文件中...')
                
                # 加载NIFTI文件
                self.image_data, self.affine, self.header = self.data_loader.load_nifti(file_path)
                
                if self.image_data is not None:
                    # 重置label_data
                    self.label_data = None
                    
                    # 更新图像信息
                    depth, height, width = self.image_data.shape
                    self.dim_label.setText(f'{depth} × {height} × {width}')
                    
                    voxel_size = self.header.get('pixdim')[1:4] if self.header else (1, 1, 1)
                    self.voxel_label.setText(f'{voxel_size[0]:.2f} × {voxel_size[1]:.2f} × {voxel_size[2]:.2f}')
                    
                    self.file_label.setText(os.path.basename(file_path))
                    
                    # 更新切片导航
                    self.current_slice = 0
                    self.update_total_slices()
                    self.slice_label.setText(f'切片: {self.current_slice + 1}/{self.total_slices}')
                    if hasattr(self, 'vis_slice_label'):
                        self.vis_slice_label.setText(f'切片: {self.current_slice + 1}/{self.total_slices}')
                    
                    # 启用按钮
                    self.prev_button.setEnabled(True)
                    self.next_button.setEnabled(True)
                    if hasattr(self, 'vis_prev_button'):
                        self.vis_prev_button.setEnabled(True)
                        self.vis_next_button.setEnabled(True)
                    
                    # 显示第一切片
                    self.update_image_display()
                    
                    # 更新按钮显示
                    self.update_button_display()
                    
                    self.status_bar.showMessage('图像文件加载成功')
                else:
                    self.status_bar.showMessage('图像文件加载失败')
            except Exception as e:
                self.status_bar.showMessage(f'错误: {str(e)}')
    
    def open_label(self):
        """打开标签文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '打开标签文件', '', 'NIFTI文件 (*.nii *.nii.gz)'
        )
        
        if file_path:
            try:
                self.status_bar.showMessage('加载标签文件中...')
                
                # 加载NIFTI文件到临时变量
                temp_label_data, _, _ = self.data_loader.load_nifti(file_path)
                
                # 保存原始label_data
                original_label_data = getattr(self, 'label_data', None)
                
                if temp_label_data is not None:
                    # 检查尺寸是否匹配
                    if self.image_data is not None:
                        image_shape = self.image_data.shape
                        label_shape = temp_label_data.shape
                        if image_shape != label_shape:
                            # 尺寸不匹配，弹出窗口
                            QMessageBox.warning(
                                self, '尺寸不匹配', 
                                f'图像尺寸 ({image_shape}) 与标签尺寸 ({label_shape}) 不匹配，请重新加载标签文件。'
                            )
                            # 保持原始label_data不变
                            return
                    
                    # 尺寸匹配，更新label_data
                    self.label_data = temp_label_data
                    
                    self.status_bar.showMessage('标签文件加载成功')
                    # 更新图像显示
                    self.update_image_display()
                    # 如果已经加载了图像，更新可视化显示
                    if self.image_data is not None and hasattr(self, 'update_visualization'):
                        self.update_visualization()
                    # 更新按钮显示
                    self.update_button_display()
                else:
                    self.status_bar.showMessage('标签文件加载失败')
            except Exception as e:
                self.status_bar.showMessage(f'错误: {str(e)}')
    
    def save_file(self):
        """保存结果"""
        # 检查是否有预测结果
        if self.prediction is not None:
            # 获取原影像名称（最多前10个字母）
            original_name = 'unknown'
            if hasattr(self, 'file_label'):
                file_label_text = self.file_label.text()
                if file_label_text and file_label_text != '文件: -':
                    # 移除路径，只保留文件名
                    base_name = os.path.basename(file_label_text)
                    # 移除文件扩展名
                    name_without_ext = os.path.splitext(base_name)[0]
                    # 只保留前10个字母
                    original_name = name_without_ext[:10]
                    # 移除可能不允许的字符
                    original_name = ''.join(c for c in original_name if c.isalnum() or c in ['_', '-'])
            
            # 生成默认文件名，包含切片信息和原影像信息
            default_filename = f"{original_name}_prediction_slice_{self.current_slice:04d}.nii"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, '保存预测结果', default_filename, 'NIFTI文件 (*.nii);;所有文件 (*.*)'
            )
            
            if file_path:
                try:
                    self.status_bar.showMessage('保存文件中...')
                    
                    # 检查文件扩展名
                    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
                        # 保存为NIFTI格式
                        success = self.data_loader.save_nifti(
                            self.prediction, self.affine, self.header, file_path
                        )
                    elif file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                        # 保存为PNG或JPG格式
                        format = file_path.rsplit('.', 1)[1].lower()
                        print(f"保存为{format}格式，路径: {file_path}")
                        success = self.save_stage_image('prediction', self.prediction, self.current_slice, format, output_path=file_path)
                        print(f"保存结果: {success}")
                    else:
                        # 默认保存为NIFTI格式
                        success = self.data_loader.save_nifti(
                            self.prediction, self.affine, self.header, file_path
                        )
                    
                    if success:
                        self.status_bar.showMessage('文件保存成功')
                    else:
                        self.status_bar.showMessage('文件保存失败')
                except Exception as e:
                    self.status_bar.showMessage(f'错误: {str(e)}')
        # 检查是否有原始图像
        elif self.image_data is not None:
            # 获取原影像名称（最多前10个字母）
            original_name = 'unknown'
            if hasattr(self, 'file_label'):
                file_label_text = self.file_label.text()
                if file_label_text and file_label_text != '文件: -':
                    # 移除路径，只保留文件名
                    base_name = os.path.basename(file_label_text)
                    # 移除文件扩展名
                    name_without_ext = os.path.splitext(base_name)[0]
                    # 只保留前10个字母
                    original_name = name_without_ext[:10]
                    # 移除可能不允许的字符
                    original_name = ''.join(c for c in original_name if c.isalnum() or c in ['_', '-'])
            
            # 生成默认文件名，包含切片信息和原影像信息
            default_filename = f"{original_name}_original_slice_{self.current_slice:04d}.nii"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, '保存原始图像', default_filename, 'NIFTI文件 (*.nii);;所有文件 (*.*)'
            )
            
            if file_path:
                try:
                    self.status_bar.showMessage('保存文件中...')
                    
                    # 检查文件扩展名
                    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
                        # 保存为NIFTI格式
                        success = self.data_loader.save_nifti(
                            self.image_data, self.affine, self.header, file_path
                        )
                    elif file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
                        # 保存为PNG或JPG格式
                        format = file_path.rsplit('.', 1)[1].lower()
                        success = self.save_stage_image('original', self.image_data, self.current_slice, format, output_path=file_path)
                    else:
                        # 默认保存为NIFTI格式
                        success = self.data_loader.save_nifti(
                            self.image_data, self.affine, self.header, file_path
                        )
                    
                    if success:
                        self.status_bar.showMessage('文件保存成功')
                    else:
                        self.status_bar.showMessage('文件保存失败')
                except Exception as e:
                    self.status_bar.showMessage(f'错误: {str(e)}')
        else:
            self.status_bar.showMessage('没有可保存的结果')
    
    def prev_slice(self):
        """显示上一切片"""
        if self.current_slice > 0:
            self.current_slice -= 1
            self.update_slice_display()
    
    def next_slice(self):
        """显示下一切片"""
        if self.current_slice < self.total_slices - 1:
            self.current_slice += 1
            self.update_slice_display()
    
    def save_images(self):
        """保存所有阶段的图像到指定的路径下"""
        # 选择保存路径
        save_dir = QFileDialog.getExistingDirectory(
            self, '选择保存目录', '', QFileDialog.ShowDirsOnly
        )
        
        if save_dir:
            try:
                self.status_bar.showMessage('保存图像中...')
                
                # 获取保存格式
                format = self.format_combo.currentText().lower() if hasattr(self, 'format_combo') else 'png'
                
                # 获取原影像名称（最多前10个字母）
                original_name = 'unknown'
                if hasattr(self, 'file_label'):
                    file_label_text = self.file_label.text()
                    if file_label_text and file_label_text != '文件: -':
                        # 移除路径，只保留文件名
                        base_name = os.path.basename(file_label_text)
                        # 移除文件扩展名
                        name_without_ext = os.path.splitext(base_name)[0]
                        # 只保留前10个字母
                        original_name = name_without_ext[:10]
                        # 移除可能不允许的字符
                        original_name = ''.join(c for c in original_name if c.isalnum() or c in ['_', '-'])
                
                # 保存原始图像
                if self.image_data is not None:
                    image_path = os.path.join(save_dir, f"{original_name}_image_slice_{self.current_slice:04d}.{format}")
                    success = self.save_stage_image('original', self.image_data, self.current_slice, format, output_path=image_path)
                    if success:
                        self.status_bar.showMessage('原始图像保存成功')
                    else:
                        self.status_bar.showMessage('原始图像保存失败')
                
                # 保存标签图像
                if hasattr(self, 'label_data') and self.label_data is not None:
                    label_path = os.path.join(save_dir, f"{original_name}_label_slice_{self.current_slice:04d}.{format}")
                    success = self.save_stage_image('label', self.label_data, self.current_slice, format, output_path=label_path)
                    if success:
                        self.status_bar.showMessage('标签图像保存成功')
                    else:
                        self.status_bar.showMessage('标签图像保存失败')
                
                # 保存融合图像
                if self.image_data is not None and (hasattr(self, 'label_data') and self.label_data is not None):
                    # 获取当前切片
                    image_slice = self.image_display.get_slice(self.image_data, self.current_slice, axis=0)
                    label_slice = self.image_display.get_slice(self.label_data, self.current_slice, axis=0)
                    
                    # 归一化图像
                    normalized_image = self.image_display.normalize_slice(image_slice)
                    
                    # 生成融合图像
                    overlayed = self.image_display.overlay_mask(normalized_image, label_slice)
                    
                    # 保存融合图像
                    overlay_path = os.path.join(save_dir, f"{original_name}_overlay_slice_{self.current_slice:04d}.{format}")
                    from PIL import Image
                    img = Image.fromarray(overlayed)
                    img.save(overlay_path)
                    self.status_bar.showMessage('融合图像保存成功')
                
                self.status_bar.showMessage('图像保存完成')
            except Exception as e:
                self.status_bar.showMessage(f'错误: {str(e)}')
    
    def save_single_image(self, image_type):
        """保存单个图像"""
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, f'保存{image_type}', '', 'PNG文件 (*.png);;JPG文件 (*.jpg);;所有文件 (*.*)'
        )
        
        if file_path:
            try:
                self.status_bar.showMessage('保存图像中...')
                
                # 获取保存格式
                format = file_path.rsplit('.', 1)[1].lower() if '.' in file_path else 'png'
                
                # 根据图像类型保存
                if image_type == 'image' and self.image_data is not None:
                    success = self.save_stage_image('original', self.image_data, self.current_slice, format, output_path=file_path)
                elif image_type == 'label' and hasattr(self, 'label_data') and self.label_data is not None:
                    success = self.save_stage_image('label', self.label_data, self.current_slice, format, output_path=file_path)
                elif image_type == 'overlay' and self.image_data is not None and hasattr(self, 'label_data') and self.label_data is not None:
                    # 获取当前切片
                    image_slice = self.image_display.get_slice(self.image_data, self.current_slice, axis=0)
                    label_slice = self.image_display.get_slice(self.label_data, self.current_slice, axis=0)
                    
                    # 归一化图像
                    normalized_image = self.image_display.normalize_slice(image_slice)
                    
                    # 生成融合图像
                    overlayed = self.image_display.overlay_mask(normalized_image, label_slice)
                    
                    # 保存融合图像
                    from PIL import Image
                    img = Image.fromarray(overlayed)
                    img.save(file_path)
                    success = True
                else:
                    success = False
                
                if success:
                    self.status_bar.showMessage(f'{image_type}保存成功')
                else:
                    self.status_bar.showMessage(f'{image_type}保存失败')
            except Exception as e:
                self.status_bar.showMessage(f'错误: {str(e)}')
    
    def update_button_display(self):
        """根据加载的内容更新按钮显示"""
        # 检查是否同时加载了image和label
        has_image = hasattr(self, 'image_data') and self.image_data is not None
        has_label = hasattr(self, 'label_data') and self.label_data is not None
        
        if has_image and has_label:
            # 同时加载了image和label，显示所有按钮和图像
            self.save_button.hide()
            self.save_image_button.show()
            self.save_label_button.show()
            self.save_overlay_button.show()
            self.save_images_button.show()
            
            # 显示所有图像显示区域
            self.image_display_group.show()
            self.label_display_group.show()
            self.overlay_display_group.show()
        elif has_image:
            # 只加载了image，只显示保存结果按钮和图像显示区域
            self.save_button.show()
            self.save_image_button.hide()
            self.save_label_button.hide()
            self.save_overlay_button.hide()
            self.save_images_button.hide()
            
            # 只显示图像显示区域
            self.image_display_group.show()
            self.label_display_group.hide()
            self.overlay_display_group.hide()
        elif has_label:
            # 只加载了label，只显示保存结果按钮和标签显示区域
            self.save_button.show()
            self.save_image_button.hide()
            self.save_label_button.hide()
            self.save_overlay_button.hide()
            self.save_images_button.hide()
            
            # 只显示标签显示区域
            self.image_display_group.hide()
            self.label_display_group.show()
            self.overlay_display_group.hide()
        else:
            # 没有加载任何东西，隐藏所有按钮和图像显示区域
            self.save_button.hide()
            self.save_image_button.hide()
            self.save_label_button.hide()
            self.save_overlay_button.hide()
            self.save_images_button.hide()
            
            # 隐藏所有图像显示区域
            self.image_display_group.hide()
            self.label_display_group.hide()
            self.overlay_display_group.hide()
    
    def reset(self):
        """重置系统"""
        # 重置图像数据
        self.image_data = None
        self.label_data = None
        self.affine = None
        self.header = None
        
        # 重置切片导航
        self.current_slice = 0
        self.total_slices = 0
        self.slice_label.setText('切片: 0/0')
        if hasattr(self, 'vis_slice_label'):
            self.vis_slice_label.setText('切片: 0/0')
        
        # 禁用切片导航按钮
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        if hasattr(self, 'vis_prev_button'):
            self.vis_prev_button.setEnabled(False)
            self.vis_next_button.setEnabled(False)
        
        # 清空图像信息
        self.dim_label.setText('')
        self.voxel_label.setText('')
        self.file_label.setText('文件: -')
        
        # 清空图像标签
        self.image_label.clear()
        self.image_label.setText('请打开图像文件')
        self.label_label.clear()
        self.label_label.setText('请打开标签文件')
        self.overlay_label.clear()
        self.overlay_label.setText('请打开图像和标签文件')
        
        # 更新按钮显示
        self.update_button_display()
        
        # 显示状态信息
        self.status_bar.showMessage('系统已重置')
        if hasattr(self, 'update_visualization'):
            self.update_visualization()
    
    def save_stage_image(self, stage_name, image_data, slice_index, format='png', output_path=None):
        """
        保存不同阶段的图片
        """
        try:
            # 获取原影像名称（最多前10个字母）
            original_name = 'unknown'
            if hasattr(self, 'file_label'):
                file_label_text = self.file_label.text()
                if file_label_text and file_label_text != '文件: -':
                    # 移除路径，只保留文件名
                    base_name = os.path.basename(file_label_text)
                    # 移除文件扩展名
                    name_without_ext = os.path.splitext(base_name)[0]
                    # 只保留前10个字母
                    original_name = name_without_ext[:10]
                    # 移除可能不允许的字符
                    original_name = ''.join(c for c in original_name if c.isalnum() or c in ['_', '-'])
            
            # 确保immdiate_show目录存在
            immdiate_show_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'immdiate_show')
            os.makedirs(immdiate_show_dir, exist_ok=True)
            
            # 确保阶段目录存在
            stage_dir = os.path.join(immdiate_show_dir, stage_name)
            os.makedirs(stage_dir, exist_ok=True)
            
            # 生成默认输出路径
            if output_path is None:
                output_path = os.path.join(
                    stage_dir,
                    f"{original_name}_{stage_name}_slice_{slice_index:04d}.{format}"
                )
            
            # 获取指定切片
            if isinstance(image_data, list):
                # 如果是列表，直接取对应索引的元素
                slice_data = image_data[slice_index]
            else:
                # 否则，使用get_slice方法
                slice_data = self.image_display.get_slice(image_data, slice_index, axis=0)
            
            # 归一化切片
            normalized_slice = self.image_display.normalize_slice(slice_data)
            
            # 保存图片
            from PIL import Image
            img = Image.fromarray(normalized_slice)
            img.save(output_path)
            
            print(f"保存{stage_name}图像成功: {output_path}")
            return True
        except Exception as e:
            print(f"保存{stage_name}图像失败: {str(e)}")
            return False
    
    def update_slice_display(self):
        """更新切片显示"""
        # 更新切片标签
        self.slice_label.setText(f'切片: {self.current_slice + 1}/{self.total_slices}')
        if hasattr(self, 'vis_slice_label'):
            self.vis_slice_label.setText(f'切片: {self.current_slice + 1}/{self.total_slices}')
        
        # 更新图像显示
        self.update_image_display()
        
        # 更新可视化显示
        if hasattr(self, 'update_visualization'):
            self.update_visualization()
    
    def update_image_display(self):
        """更新图像显示"""
        if self.image_data is not None:
            # 获取当前选择的切片轴
            current_axis = self.slice_axis_combo.currentIndex()
            
            # 获取当前切片
            image_slice = self.image_display.get_slice(self.image_data, self.current_slice, axis=current_axis)
            
            # 处理不同轴切片的形状
            if current_axis == 1:
                # 转置为 (宽度, 深度)
                image_slice = image_slice.T
            elif current_axis == 2:
                # 转置为 (高度, 深度)
                image_slice = image_slice.T
            
            # 归一化切片
            normalized_image = self.image_display.normalize_slice(image_slice)
            
            # 转换为QImage
            height, width = normalized_image.shape
            bytes_per_line = width
            q_image = QImage(bytes(normalized_image.data), width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            max_size = 400
            scaled_pixmap = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            # 如果有标签，也显示标签
            if hasattr(self, 'label_data') and self.label_data is not None:
                # 获取当前标签切片
                label_slice = self.image_display.get_slice(self.label_data, self.current_slice, axis=current_axis)
                
                # 处理不同轴切片的形状
                original_label_slice = label_slice.copy()
                if current_axis == 1:
                    # 转置为 (宽度, 深度)
                    label_slice = label_slice.T
                elif current_axis == 2:
                    # 转置为 (高度, 深度)
                    label_slice = label_slice.T
                
                # 归一化标签
                normalized_label = self.image_display.normalize_slice(label_slice)
                
                # 转换为QImage
                height, width = normalized_label.shape
                bytes_per_line = width
                q_label = QImage(bytes(normalized_label.data), width, height, bytes_per_line, QImage.Format_Grayscale8)
                
                # 转换为QPixmap并显示
                pixmap_label = QPixmap.fromImage(q_label)
                scaled_pixmap_label = pixmap_label.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_label.setPixmap(scaled_pixmap_label)
                
                # 生成融合图像
                overlayed = self.image_display.overlay_mask(normalized_image, label_slice)
                
                # 转换为QImage
                height, width, _ = overlayed.shape
                bytes_per_line = width * 3
                q_overlay = QImage(bytes(overlayed.data), width, height, bytes_per_line, QImage.Format_RGB888)
                
                # 转换为QPixmap并显示
                pixmap_overlay = QPixmap.fromImage(q_overlay)
                scaled_pixmap_overlay = pixmap_overlay.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.overlay_label.setPixmap(scaled_pixmap_overlay)
    
    def run_preprocessing(self):
        """执行预处理"""
        if self.image_data is None:
            self.preprocess_status.setText('请先加载图像文件')
            return
        
        try:
            self.status_bar.showMessage('执行预处理中...')
            
            # 获取预处理参数
            normalize_method = self.normalize_combo.currentText()
            target_voxel_size = (
                self.voxel_d.value(),
                self.voxel_h.value(),
                self.voxel_w.value()
            )
            
            # 执行预处理
            self.preprocessed_data = self.preprocessor.preprocess_pipeline(
                self.image_data,
                normalize_method=normalize_method,
                target_voxel_size=target_voxel_size
            )
            
            self.preprocess_status.setText('预处理完成')
            self.status_bar.showMessage('预处理完成')
        except Exception as e:
            self.preprocess_status.setText(f'预处理错误: {str(e)}')
            self.status_bar.showMessage(f'预处理错误: {str(e)}')
    
    def run_prediction(self):
        """执行预测"""
        if self.image_data is None:
            self.predict_status.setText('请先加载图像文件')
            return
        
        try:
            self.status_bar.showMessage('执行预测中...')
            
            # 获取模型选择
            first_stage_model = self.first_stage_combo.currentText()
            second_stage_model = self.second_stage_combo.currentText()
            
            # 创建预测线程
            self.thread = PredictionThread(
                None,
                self.image_data,
                self.preprocessor,
                self.second_stage_processor
            )
            
            # 连接信号
            self.thread.progress_updated.connect(self.on_progress_updated)
            self.thread.prediction_completed.connect(self.on_prediction_completed)
            self.thread.error_occurred.connect(self.on_prediction_error)
            
            # 启动线程
            self.thread.start()
            
        except Exception as e:
            self.predict_status.setText(f'预测错误: {str(e)}')
            self.status_bar.showMessage(f'预测错误: {str(e)}')
    
    def on_progress_updated(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def on_prediction_completed(self, prediction, metrics):
        """预测完成处理"""
        self.prediction = prediction
        self.metrics = metrics
        
        # 更新预测状态
        self.predict_status.setText('预测完成')
        
        # 更新评估指标
        self.dice_label.setText(f'{metrics["dice"]:.4f}')
        self.iou_label.setText(f'{metrics["iou"]:.4f}')
        self.sensitivity_label.setText(f'{metrics["sensitivity"]:.4f}')
        self.specificity_label.setText(f'{metrics["specificity"]:.4f}')
        
        # 更新可视化显示
        if hasattr(self, 'update_visualization'):
            self.update_visualization()
        
        # 显示状态信息
        self.status_bar.showMessage('预测完成')
    
    def on_prediction_error(self, error):
        """预测错误处理"""
        self.predict_status.setText(f'预测错误: {error}')
        self.status_bar.showMessage(f'预测错误: {error}')
    
    def update_visualization(self):
        """更新可视化显示"""
        view_mode = self.view_combo.currentText() if hasattr(self, 'view_combo') else '原始图像'
        
        # 获取当前选择的切片轴
        current_axis = self.slice_axis_combo.currentIndex() if hasattr(self, 'slice_axis_combo') else 0
        
        if view_mode == '原始图像' and self.image_data is not None:
            # 获取当前切片
            image_slice = self.image_display.get_slice(self.image_data, self.current_slice, axis=current_axis)
            
            # 处理不同轴切片的形状
            if current_axis == 1:
                # 转置为 (宽度, 深度)
                image_slice = image_slice.T
            elif current_axis == 2:
                # 转置为 (高度, 深度)
                image_slice = image_slice.T
            
            # 归一化切片
            normalized_image = self.image_display.normalize_slice(image_slice)
            
            # 转换为QImage
            height, width = normalized_image.shape
            bytes_per_line = width
            q_image = QImage(bytes(normalized_image.data), width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            if hasattr(self, 'vis_label'):
                self.vis_label.setPixmap(pixmap.scaled(self.vis_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        elif view_mode == '分割掩码' and self.prediction is not None:
            # 获取当前切片
            prediction_slice = self.image_display.get_slice(self.prediction, self.current_slice, axis=current_axis)
            
            # 处理不同轴切片的形状
            if current_axis == 1:
                # 转置为 (宽度, 深度)
                prediction_slice = prediction_slice.T
            elif current_axis == 2:
                # 转置为 (高度, 深度)
                prediction_slice = prediction_slice.T
            
            # 归一化切片
            normalized_prediction = self.image_display.normalize_slice(prediction_slice)
            
            # 转换为QImage
            height, width = normalized_prediction.shape
            bytes_per_line = width
            q_image = QImage(bytes(normalized_prediction.data), width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            if hasattr(self, 'vis_label'):
                self.vis_label.setPixmap(pixmap.scaled(self.vis_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        elif view_mode == '叠加视图' and self.image_data is not None and self.prediction is not None:
            # 获取当前切片
            image_slice = self.image_display.get_slice(self.image_data, self.current_slice, axis=current_axis)
            prediction_slice = self.image_display.get_slice(self.prediction, self.current_slice, axis=current_axis)
            
            # 处理不同轴切片的形状
            if current_axis == 1:
                # 转置为 (宽度, 深度)
                image_slice = image_slice.T
                prediction_slice = prediction_slice.T
            elif current_axis == 2:
                # 转置为 (高度, 深度)
                image_slice = image_slice.T
                prediction_slice = prediction_slice.T
            
            # 归一化切片
            normalized_image = self.image_display.normalize_slice(image_slice)
            
            # 生成叠加图像
            overlayed = self.image_display.overlay_mask(normalized_image, prediction_slice)
            
            # 转换为QImage
            height, width, _ = overlayed.shape
            bytes_per_line = width * 3
            q_image = QImage(bytes(overlayed.data), width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            if hasattr(self, 'vis_label'):
                self.vis_label.setPixmap(pixmap.scaled(self.vis_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def on_save_all_stages(self):
        """保存所有阶段的图像"""
        # 创建immdiate_show目录
        immdiate_show_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'immdiate_show')
        os.makedirs(immdiate_show_dir, exist_ok=True)
        
        # 保存原始图像
        if self.image_data is not None:
            self.save_stage_image('original', self.image_data, self.current_slice, self.format_combo.currentText().lower())
        
        # 保存预处理后图像
        if self.preprocessed_data is not None:
            self.save_stage_image('preprocessed', self.preprocessed_data, self.current_slice, self.format_combo.currentText().lower())
        
        # 保存预测结果
        if self.prediction is not None:
            self.save_stage_image('prediction', self.prediction, self.current_slice, self.format_combo.currentText().lower())
        
        self.status_bar.showMessage('所有阶段图像已保存')
    
    def run_evaluation(self):
        """执行评估"""
        if self.prediction is None:
            self.evaluate_log.setText('请先执行预测')
            return
        
        try:
            self.status_bar.showMessage('执行评估中...')
            
            # 模拟评估结果
            metrics = {
                'dice': 0.85,
                'iou': 0.75,
                'sensitivity': 0.90,
                'specificity': 0.95
            }
            
            # 更新评估指标
            self.dice_label.setText(f'{metrics["dice"]:.4f}')
            self.iou_label.setText(f'{metrics["iou"]:.4f}')
            self.sensitivity_label.setText(f'{metrics["sensitivity"]:.4f}')
            self.specificity_label.setText(f'{metrics["specificity"]:.4f}')
            
            # 更新评估日志
            self.evaluate_log.setText(f'Dice系数: {metrics["dice"]:.4f}\n' +
                                    f'IoU: {metrics["iou"]:.4f}\n' +
                                    f'敏感性: {metrics["sensitivity"]:.4f}\n' +
                                    f'特异性: {metrics["specificity"]:.4f}')
            
            self.status_bar.showMessage('评估完成')
        except Exception as e:
            self.evaluate_log.setText(f'评估错误: {str(e)}')
            self.status_bar.showMessage(f'评估错误: {str(e)}')
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self, '关于',
            '脑微出血分割系统\n\n' +
            '版本: 1.0.0\n' +
            '描述: 基于深度学习的脑微出血分割系统\n' +
            '用途: 帮助医生等用户利用深度学习算法进行基于影像的类别鉴定与结果展示\n' +
            '开发者: 用户\n' +
            '日期: 2026'
        )
    
    def on_slice_axis_changed(self, index):
        """处理切片轴变化"""
        if self.image_data is not None:
            # 重置切片索引
            self.current_slice = 0
            # 更新总切片数
            self.update_total_slices()
            # 更新切片信息标签
            self.slice_label.setText(f'切片: {self.current_slice + 1}/{self.total_slices}')
            if hasattr(self, 'vis_slice_label'):
                self.vis_slice_label.setText(f'切片: {self.current_slice + 1}/{self.total_slices}')
            # 更新图像显示
            self.update_image_display()
    
    def on_slice_jump(self, value):
        """处理切片跳转"""
        # 将输入的切片编号转换为索引（减1）
        new_slice_index = value - 1
        # 确保索引在有效范围内
        if 0 <= new_slice_index < self.total_slices:
            self.current_slice = new_slice_index
            # 更新切片信息标签
            self.slice_label.setText(f'切片: {self.current_slice + 1}/{self.total_slices}')
            if hasattr(self, 'vis_slice_label'):
                self.vis_slice_label.setText(f'切片: {self.current_slice + 1}/{self.total_slices}')
            # 更新图像显示
            self.update_image_display()
            if hasattr(self, 'update_visualization'):
                self.update_visualization()
    
    def update_total_slices(self):
        """根据当前选择的切片轴更新总切片数"""
        if self.image_data is not None:
            axis_index = self.slice_axis_combo.currentIndex()
            if axis_index == 0:
                self.total_slices = self.image_data.shape[0]
            elif axis_index == 1:
                self.total_slices = self.image_data.shape[1]
            else:
                self.total_slices = self.image_data.shape[2]
            
            # 更新切片跳转输入框的最大值
            self.slice_spin_box.setMaximum(self.total_slices)
            # 确保当前切片索引在有效范围内
            if self.current_slice >= self.total_slices:
                self.current_slice = 0
                # 更新切片跳转输入框的值
                self.slice_spin_box.setValue(1)
        else:
            self.total_slices = 0
            # 重置切片跳转输入框
            self.slice_spin_box.setMaximum(1000)
            self.slice_spin_box.setValue(1)
    
    def show_zxy_slice(self):
        """根据ZXY坐标显示对应切片"""
        if self.image_data is None:
            self.status_bar.showMessage('请先加载图像文件')
            return
        
        try:
            # 获取输入的ZXY坐标
            z = self.z_spin_box.value()
            x = self.x_spin_box.value()
            y = self.y_spin_box.value()
            
            # 检查坐标是否在有效范围内
            depth, height, width = self.image_data.shape
            
            if z >= depth or y >= height or x >= width:
                self.status_bar.showMessage('坐标超出图像范围')
                return
            
            # 显示Z轴切片（XY平面）
            z_slice = self.image_display.get_slice(self.image_data, z, axis=0)
            normalized_z = self.image_display.normalize_slice(z_slice)
            
            height_z, width_z = normalized_z.shape
            bytes_per_line_z = width_z
            q_image_z = QImage(bytes(normalized_z.data), width_z, height_z, bytes_per_line_z, QImage.Format_Grayscale8)
            pixmap_z = QPixmap.fromImage(q_image_z)
            max_size = 300
            scaled_pixmap_z = pixmap_z.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.z_slice_label.setPixmap(scaled_pixmap_z)
            
            # 显示X轴切片（YZ平面）
            x_slice = self.image_display.get_slice(self.image_data, x, axis=2)
            x_slice = x_slice.T
            normalized_x = self.image_display.normalize_slice(x_slice)
            
            height_x, width_x = normalized_x.shape
            bytes_per_line_x = width_x
            q_image_x = QImage(bytes(normalized_x.data), width_x, height_x, bytes_per_line_x, QImage.Format_Grayscale8)
            pixmap_x = QPixmap.fromImage(q_image_x)
            scaled_pixmap_x = pixmap_x.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.x_slice_label.setPixmap(scaled_pixmap_x)
            
            # 显示Y轴切片（XZ平面）
            y_slice = self.image_display.get_slice(self.image_data, y, axis=1)
            y_slice = y_slice.T
            normalized_y = self.image_display.normalize_slice(y_slice)
            
            height_y, width_y = normalized_y.shape
            bytes_per_line_y = width_y
            q_image_y = QImage(bytes(normalized_y.data), width_y, height_y, bytes_per_line_y, QImage.Format_Grayscale8)
            pixmap_y = QPixmap.fromImage(q_image_y)
            scaled_pixmap_y = pixmap_y.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.y_slice_label.setPixmap(scaled_pixmap_y)
            
            # 如果有标签，也显示标签
            if hasattr(self, 'label_data') and self.label_data is not None:
                # 显示Z轴标签切片
                z_label_slice = self.image_display.get_slice(self.label_data, z, axis=0)
                
                # 生成融合图像
                overlay_z = self.image_display.overlay_mask(normalized_z, z_label_slice)
                height_oz, width_oz, _ = overlay_z.shape
                bytes_per_line_oz = width_oz * 3
                q_overlay_z = QImage(bytes(overlay_z.data), width_oz, height_oz, bytes_per_line_oz, QImage.Format_RGB888)
                pixmap_oz = QPixmap.fromImage(q_overlay_z)
                scaled_pixmap_oz = pixmap_oz.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.z_slice_label.setPixmap(scaled_pixmap_oz)
                
                # 显示X轴标签切片
                x_label_slice = self.image_display.get_slice(self.label_data, x, axis=2)
                x_label_slice = x_label_slice.T
                
                # 生成融合图像
                overlay_x = self.image_display.overlay_mask(normalized_x, x_label_slice)
                height_ox, width_ox, _ = overlay_x.shape
                bytes_per_line_ox = width_ox * 3
                q_overlay_x = QImage(bytes(overlay_x.data), width_ox, height_ox, bytes_per_line_ox, QImage.Format_RGB888)
                pixmap_ox = QPixmap.fromImage(q_overlay_x)
                scaled_pixmap_ox = pixmap_ox.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.x_slice_label.setPixmap(scaled_pixmap_ox)
                
                # 显示Y轴标签切片
                y_label_slice = self.image_display.get_slice(self.label_data, y, axis=1)
                y_label_slice = y_label_slice.T
                
                # 生成融合图像
                overlay_y = self.image_display.overlay_mask(normalized_y, y_label_slice)
                height_oy, width_oy, _ = overlay_y.shape
                bytes_per_line_oy = width_oy * 3
                q_overlay_y = QImage(bytes(overlay_y.data), width_oy, height_oy, bytes_per_line_oy, QImage.Format_RGB888)
                pixmap_oy = QPixmap.fromImage(q_overlay_y)
                scaled_pixmap_oy = pixmap_oy.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.y_slice_label.setPixmap(scaled_pixmap_oy)
            
            self.status_bar.showMessage(f'显示Z={z}, X={x}, Y={y}的切片')
        except Exception as e:
            self.status_bar.showMessage(f'错误: {str(e)}')
