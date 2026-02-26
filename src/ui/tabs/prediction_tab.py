from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, 
    QLabel, QComboBox, QGroupBox, QFormLayout, QProgressBar, QTextEdit
)


def create_prediction_tab(parent):
    """创建模型预测选项卡"""
    prediction_tab = QWidget()
    parent.tab_widget.addTab(prediction_tab, '模型预测')
    
    # 创建布局
    layout = QVBoxLayout(prediction_tab)
    
    # 创建模型选择组
    model_group = QGroupBox('模型选择')
    layout.addWidget(model_group)
    
    model_layout = QFormLayout(model_group)
    
    # 一阶段模型选择
    parent.first_stage_combo = QComboBox()
    parent.first_stage_combo.addItems(['3D U-Net', '2D U-Net'])
    model_layout.addRow('一阶段模型:', parent.first_stage_combo)
    
    # 二阶段模型选择
    parent.second_stage_combo = QComboBox()
    parent.second_stage_combo.addItems(['精细分割模型', '边缘优化模型'])
    model_layout.addRow('二阶段模型:', parent.second_stage_combo)
    
    # 预测按钮
    parent.predict_button = QPushButton('执行预测')
    parent.predict_button.clicked.connect(parent.run_prediction)
    layout.addWidget(parent.predict_button)
    
    # 进度条
    parent.progress_bar = QProgressBar()
    parent.progress_bar.setValue(0)
    layout.addWidget(parent.progress_bar)
    
    # 预测状态
    parent.predict_status = QTextEdit()
    parent.predict_status.setReadOnly(True)
    parent.predict_status.setPlaceholderText('预测结果将显示在这里')
    layout.addWidget(parent.predict_status)
