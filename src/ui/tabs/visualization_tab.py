from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QComboBox, QGroupBox, QFileDialog, QLineEdit, QCheckBox, QGridLayout
)
from PyQt5.QtCore import Qt


def create_visualization_tab(parent):
    """创建结果可视化选项卡"""
    visualization_tab = QWidget()
    parent.tab_widget.addTab(visualization_tab, '结果可视化')
    
    # 创建布局
    layout = QVBoxLayout(visualization_tab)
    
    # 创建文件选择组
    file_group = QGroupBox('文件选择')
    layout.addWidget(file_group)
    
    file_layout = QVBoxLayout(file_group)
    file_layout.setSpacing(5)
    
    # SWI影像选择
    swi_layout = QHBoxLayout()
    swi_layout.addWidget(QLabel('SWI影像:'))
    parent.swi_file_path = QLineEdit()
    parent.swi_file_path.setMinimumWidth(400)
    swi_layout.addWidget(parent.swi_file_path)
    swi_button = QPushButton('浏览')
    swi_button.setFixedWidth(60)
    swi_button.clicked.connect(lambda: browse_file(parent, 'swi_file_path'))
    swi_layout.addWidget(swi_button)
    file_layout.addLayout(swi_layout)
    
    # GroundTruth选择
    gt_layout = QHBoxLayout()
    gt_layout.addWidget(QLabel('GroundTruth:'))
    parent.gt_file_path = QLineEdit()
    parent.gt_file_path.setMinimumWidth(400)
    gt_layout.addWidget(parent.gt_file_path)
    gt_button = QPushButton('浏览')
    gt_button.setFixedWidth(60)
    gt_button.clicked.connect(lambda: browse_file(parent, 'gt_file_path'))
    gt_layout.addWidget(gt_button)
    file_layout.addLayout(gt_layout)
    
    # 预测Mask选择
    mask_layout = QHBoxLayout()
    mask_layout.addWidget(QLabel('预测Mask:'))
    parent.mask_file_path = QLineEdit()
    parent.mask_file_path.setMinimumWidth(400)
    mask_layout.addWidget(parent.mask_file_path)
    mask_button = QPushButton('浏览')
    mask_button.setFixedWidth(60)
    mask_button.clicked.connect(lambda: browse_file(parent, 'mask_file_path'))
    mask_layout.addWidget(mask_button)
    file_layout.addLayout(mask_layout)
    
    # 加载按钮
    load_button = QPushButton('加载文件')
    load_button.setFixedWidth(100)
    load_button.clicked.connect(lambda: load_visualization_files(parent))
    file_layout.addWidget(load_button, alignment=Qt.AlignCenter)
    
    # 创建可视化控制组
    control_group = QGroupBox('可视化控制')
    layout.addWidget(control_group)
    
    control_layout = QVBoxLayout(control_group)
    
    # 切片导航和轴选择
    slice_control_layout = QHBoxLayout()
    
    # 切片轴选择
    axis_layout = QHBoxLayout()
    axis_layout.addWidget(QLabel('切片轴:'))
    parent.vis_axis_combo = QComboBox()
    parent.vis_axis_combo.addItems(['Z轴', 'X轴', 'Y轴'])
    parent.vis_axis_combo.currentIndexChanged.connect(lambda: on_axis_changed(parent))
    axis_layout.addWidget(parent.vis_axis_combo)
    slice_control_layout.addLayout(axis_layout)
    
    # 切片导航
    vis_slice_layout = QHBoxLayout()
    parent.vis_prev_button = QPushButton('上一切片')
    parent.vis_prev_button.clicked.connect(parent.prev_slice)
    parent.vis_prev_button.setEnabled(False)
    vis_slice_layout.addWidget(parent.vis_prev_button)
    
    parent.vis_slice_label = QLabel('切片: 0/0')
    vis_slice_layout.addWidget(parent.vis_slice_label)
    
    parent.vis_next_button = QPushButton('下一切片')
    parent.vis_next_button.clicked.connect(parent.next_slice)
    parent.vis_next_button.setEnabled(False)
    vis_slice_layout.addWidget(parent.vis_next_button)
    
    slice_control_layout.addLayout(vis_slice_layout)
    control_layout.addLayout(slice_control_layout)
    
    # 图像选择复选框
    image_selection_layout = QHBoxLayout()
    image_selection_layout.addWidget(QLabel('选择要显示的图像:'))
    
    # 图像选择复选框
    parent.vis_image_checkbox = QCheckBox('原图像')
    parent.vis_image_checkbox.setChecked(True)
    parent.vis_image_checkbox.stateChanged.connect(parent.update_visualization)
    image_selection_layout.addWidget(parent.vis_image_checkbox)
    
    parent.vis_gt_checkbox = QCheckBox('GroundTruth')
    parent.vis_gt_checkbox.setChecked(True)
    parent.vis_gt_checkbox.stateChanged.connect(parent.update_visualization)
    image_selection_layout.addWidget(parent.vis_gt_checkbox)
    
    parent.vis_mask_checkbox = QCheckBox('预测Mask')
    parent.vis_mask_checkbox.setChecked(True)
    parent.vis_mask_checkbox.stateChanged.connect(parent.update_visualization)
    image_selection_layout.addWidget(parent.vis_mask_checkbox)
    
    parent.vis_image_gt_checkbox = QCheckBox('原图像+GT')
    parent.vis_image_gt_checkbox.setChecked(True)
    parent.vis_image_gt_checkbox.stateChanged.connect(parent.update_visualization)
    image_selection_layout.addWidget(parent.vis_image_gt_checkbox)
    
    parent.vis_image_mask_checkbox = QCheckBox('原图像+Mask')
    parent.vis_image_mask_checkbox.setChecked(True)
    parent.vis_image_mask_checkbox.stateChanged.connect(parent.update_visualization)
    image_selection_layout.addWidget(parent.vis_image_mask_checkbox)
    
    parent.vis_gt_mask_checkbox = QCheckBox('GT+Mask')
    parent.vis_gt_mask_checkbox.setChecked(True)
    parent.vis_gt_mask_checkbox.stateChanged.connect(parent.update_visualization)
    image_selection_layout.addWidget(parent.vis_gt_mask_checkbox)
    
    control_layout.addLayout(image_selection_layout)
    
    # 保存格式选择和按钮
    save_layout = QHBoxLayout()
    parent.format_combo = QComboBox()
    parent.format_combo.addItems(['PNG', 'JPG'])
    save_layout.addWidget(parent.format_combo)
    
    parent.save_stages_button = QPushButton('保存所有阶段')
    parent.save_stages_button.clicked.connect(parent.on_save_all_stages)
    save_layout.addWidget(parent.save_stages_button)
    
    control_layout.addLayout(save_layout)
    
    # 创建可视化显示区域
    vis_group = QGroupBox('结果展示')
    layout.addWidget(vis_group)
    
    # 使用垂直布局，内部使用水平布局来放置图像
    vis_layout = QVBoxLayout(vis_group)
    
    # 创建一个容器来放置图像行
    parent.vis_container = QWidget()
    parent.vis_container_layout = QVBoxLayout(parent.vis_container)
    vis_layout.addWidget(parent.vis_container)
    
    # 创建6个图像显示标签
    parent.vis_labels = []
    parent.vis_titles = [
        '1. 原图像',
        '2. GroundTruth',
        '3. 预测Mask',
        '4. 原图像+GT',
        '5. 原图像+Mask',
        '6. GT+Mask'
    ]
    
    # 初始创建所有标签，但默认隐藏
    for i in range(6):
        # 创建图像显示标签
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setText('请加载文件')
        img_label.hide()  # 默认隐藏
        parent.vis_labels.append(img_label)


def browse_file(parent, attribute_name):
    """浏览文件"""
    file_path, _ = QFileDialog.getOpenFileName(
        None, "选择文件", "", "NIfTI文件 (*.nii *.nii.gz);;所有文件 (*.*)"
    )
    if file_path:
        getattr(parent, attribute_name).setText(file_path)


def load_visualization_files(parent):
    """加载可视化文件"""
    swi_path = parent.swi_file_path.text()
    gt_path = parent.gt_file_path.text()
    mask_path = parent.mask_file_path.text()
    
    if not swi_path or not gt_path or not mask_path:
        return
    
    try:
        import nibabel as nib
        import numpy as np
        
        # 加载SWI影像
        swi_img = nib.load(swi_path)
        parent.vis_image_data = swi_img.get_fdata()
        
        # 加载GroundTruth
        gt_img = nib.load(gt_path)
        parent.vis_gt_data = gt_img.get_fdata()
        
        # 加载预测Mask
        mask_img = nib.load(mask_path)
        parent.vis_mask_data = mask_img.get_fdata()
        
        # 更新切片信息
        update_slice_info(parent)
        
        # 启用切片导航按钮
        parent.vis_prev_button.setEnabled(parent.total_slices > 1)
        parent.vis_next_button.setEnabled(parent.total_slices > 1)
        
        # 更新可视化
        parent.update_visualization()
        
    except Exception as e:
        print(f'加载文件错误: {str(e)}')

def update_slice_info(parent):
    """更新切片信息"""
    if not hasattr(parent, 'vis_image_data') or parent.vis_image_data is None:
        return
    
    # 获取当前轴
    current_axis = parent.vis_axis_combo.currentIndex() if hasattr(parent, 'vis_axis_combo') else 0
    
    # 获取对应轴的切片数量
    data_shape = parent.vis_image_data.shape
    parent.total_slices = data_shape[current_axis]
    parent.current_slice = 0
    
    # 更新切片标签
    parent.vis_slice_label.setText(f'切片: {parent.current_slice + 1}/{parent.total_slices}')

def on_axis_changed(parent):
    """轴改变时的处理"""
    update_slice_info(parent)
    parent.update_visualization()
    # 更新按钮状态
    parent.vis_prev_button.setEnabled(parent.total_slices > 1)
    parent.vis_next_button.setEnabled(parent.total_slices > 1)
