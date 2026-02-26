from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QComboBox, QGroupBox
)
from PyQt5.QtCore import Qt


def create_visualization_tab(parent):
    """创建结果可视化选项卡"""
    visualization_tab = QWidget()
    parent.tab_widget.addTab(visualization_tab, '结果可视化')
    
    # 创建布局
    layout = QVBoxLayout(visualization_tab)
    
    # 创建可视化控制组
    control_group = QGroupBox('可视化控制')
    layout.addWidget(control_group)
    
    control_layout = QHBoxLayout(control_group)
    
    # 视图选择
    parent.view_combo = QComboBox()
    parent.view_combo.addItems(['原始图像', '分割掩码', '叠加视图'])
    parent.view_combo.currentIndexChanged.connect(parent.update_visualization)
    control_layout.addWidget(parent.view_combo)
    
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
    
    control_layout.addLayout(vis_slice_layout)
    
    # 保存格式选择
    parent.format_combo = QComboBox()
    parent.format_combo.addItems(['PNG', 'JPG'])
    control_layout.addWidget(parent.format_combo)
    
    # 保存按钮
    parent.save_stages_button = QPushButton('保存所有阶段')
    parent.save_stages_button.clicked.connect(parent.on_save_all_stages)
    control_layout.addWidget(parent.save_stages_button)
    
    # 创建可视化显示区域
    vis_group = QGroupBox('结果展示')
    layout.addWidget(vis_group)
    
    vis_layout = QVBoxLayout(vis_group)
    
    parent.vis_label = QLabel()
    parent.vis_label.setAlignment(Qt.AlignCenter)
    parent.vis_label.setText('预测结果将显示在这里')
    vis_layout.addWidget(parent.vis_label)
