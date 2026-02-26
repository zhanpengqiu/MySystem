from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, 
    QLabel, QGroupBox, QFormLayout, QTextEdit
)


def create_evaluation_tab(parent):
    """创建评估选项卡"""
    evaluation_tab = QWidget()
    parent.tab_widget.addTab(evaluation_tab, '评估')
    
    # 创建布局
    layout = QVBoxLayout(evaluation_tab)
    
    # 创建评估指标组
    metrics_group = QGroupBox('评估指标')
    layout.addWidget(metrics_group)
    
    metrics_layout = QFormLayout(metrics_group)
    
    parent.dice_label = QLabel('0.0000')
    metrics_layout.addRow('Dice系数:', parent.dice_label)
    
    parent.iou_label = QLabel('0.0000')
    metrics_layout.addRow('IoU:', parent.iou_label)
    
    parent.sensitivity_label = QLabel('0.0000')
    metrics_layout.addRow('敏感性:', parent.sensitivity_label)
    
    parent.specificity_label = QLabel('0.0000')
    metrics_layout.addRow('特异性:', parent.specificity_label)
    
    # 创建评估按钮
    parent.evaluate_button = QPushButton('执行评估')
    parent.evaluate_button.clicked.connect(parent.run_evaluation)
    layout.addWidget(parent.evaluate_button)
    
    # 评估日志
    parent.evaluate_log = QTextEdit()
    parent.evaluate_log.setReadOnly(True)
    parent.evaluate_log.setPlaceholderText('评估结果将显示在这里')
    layout.addWidget(parent.evaluate_log)
