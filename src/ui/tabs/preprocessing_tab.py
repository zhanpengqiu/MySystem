from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSpinBox, QComboBox, QGroupBox, QFormLayout, QTextEdit
)


def create_preprocessing_tab(parent):
    """创建预处理选项卡"""
    preprocess_tab = QWidget()
    parent.tab_widget.addTab(preprocess_tab, '预处理')
    
    # 创建布局
    layout = QVBoxLayout(preprocess_tab)
    
    # 创建预处理参数组
    param_group = QGroupBox('预处理参数')
    layout.addWidget(param_group)
    
    param_layout = QFormLayout(param_group)
    
    # 归一化方法选择
    parent.normalize_combo = QComboBox()
    parent.normalize_combo.addItems(['z-score', 'histogram'])
    param_layout.addRow('归一化方法:', parent.normalize_combo)
    
    # 目标体素大小
    voxel_layout = QHBoxLayout()
    parent.voxel_d = QSpinBox()
    parent.voxel_d.setRange(1, 10)
    parent.voxel_d.setValue(1)
    voxel_layout.addWidget(parent.voxel_d)
    
    parent.voxel_h = QSpinBox()
    parent.voxel_h.setRange(1, 10)
    parent.voxel_h.setValue(1)
    voxel_layout.addWidget(parent.voxel_h)
    
    parent.voxel_w = QSpinBox()
    parent.voxel_w.setRange(1, 10)
    parent.voxel_w.setValue(1)
    voxel_layout.addWidget(parent.voxel_w)
    
    param_layout.addRow('目标体素大小 (d, h, w):', voxel_layout)
    
    # 切片轴选择
    parent.axis_combo = QComboBox()
    parent.axis_combo.addItems(['深度轴 (z)', '高度轴 (y)', '宽度轴 (x)'])
    param_layout.addRow('切片轴:', parent.axis_combo)
    
    # 预处理按钮
    parent.preprocess_button = QPushButton('执行预处理')
    parent.preprocess_button.clicked.connect(parent.run_preprocessing)
    layout.addWidget(parent.preprocess_button)
    
    # 预处理状态
    parent.preprocess_status = QTextEdit()
    parent.preprocess_status.setReadOnly(True)
    parent.preprocess_status.setPlaceholderText('预处理结果将显示在这里')
    layout.addWidget(parent.preprocess_status)
