from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QRadioButton, 
    QLabel, QComboBox, QGroupBox, QFormLayout, QProgressBar, QTextEdit,
    QFileDialog, QLineEdit, QCheckBox
)


def set_prediction_mode(parent, stage_prefix, mode):
    """设置预测模式"""
    # 确保只有一个按钮被选中
    if mode == 'single':
        getattr(parent, f'{stage_prefix}_single_radio').setChecked(True)
        getattr(parent, f'{stage_prefix}_batch_radio').setChecked(False)
    else:
        getattr(parent, f'{stage_prefix}_single_radio').setChecked(False)
        getattr(parent, f'{stage_prefix}_batch_radio').setChecked(True)


def browse_file_or_folder(parent, stage_prefix):
    """浏览选择文件或文件夹"""
    # 检查当前预测模式
    is_single = getattr(parent, f'{stage_prefix}_single_radio').isChecked()
    
    if is_single:
        # 单个文件选择
        file_path, _ = QFileDialog.getOpenFileName(
            None, "选择SWI影像文件", "", "NIFTI文件 (*.nii *.nii.gz);;所有文件 (*)"
        )
        if file_path:
            getattr(parent, f'{stage_prefix}_file_path').setText(file_path)
    else:
        # 文件夹选择
        folder_path = QFileDialog.getExistingDirectory(
            None, "选择SWI影像文件夹"
        )
        if folder_path:
            getattr(parent, f'{stage_prefix}_file_path').setText(folder_path)


def create_prediction_tab(parent):
    """创建模型预测选项卡"""
    prediction_tab = QWidget()
    parent.tab_widget.addTab(prediction_tab, '模型预测')
    
    # 创建布局
    layout = QVBoxLayout(prediction_tab)
    
    # 一阶段模型预测
    first_stage_group = QGroupBox('一阶段模型预测')
    layout.addWidget(first_stage_group)
    
    first_stage_layout = QVBoxLayout(first_stage_group)
    
    # 一阶段模型选择
    first_stage_model_layout = QFormLayout()
    parent.first_stage_combo = QComboBox()
    parent.first_stage_combo.addItems(['3D U-Net', '2D U-Net'])
    first_stage_model_layout.addRow('一阶段模型:', parent.first_stage_combo)
    first_stage_layout.addLayout(first_stage_model_layout)
    
    # 预测模式选择
    from PyQt5.QtWidgets import QButtonGroup
    first_stage_pred_mode_layout = QHBoxLayout()
    
    # 创建按钮组
    parent.first_stage_mode_group = QButtonGroup()
    
    parent.first_stage_single_radio = QRadioButton('单个文件预测')
    parent.first_stage_single_radio.setChecked(True)
    parent.first_stage_mode_group.addButton(parent.first_stage_single_radio)
    first_stage_pred_mode_layout.addWidget(parent.first_stage_single_radio)
    
    parent.first_stage_batch_radio = QRadioButton('批量文件夹预测')
    parent.first_stage_mode_group.addButton(parent.first_stage_batch_radio)
    first_stage_pred_mode_layout.addWidget(parent.first_stage_batch_radio)
    
    # 连接信号
    parent.first_stage_single_radio.clicked.connect(lambda: set_prediction_mode(parent, 'first_stage', 'single'))
    parent.first_stage_batch_radio.clicked.connect(lambda: set_prediction_mode(parent, 'first_stage', 'batch'))
    
    first_stage_layout.addLayout(first_stage_pred_mode_layout)
    
    # 文件选择
    first_stage_file_layout = QFormLayout()
    parent.first_stage_file_path = QLineEdit()
    parent.first_stage_file_path.setPlaceholderText('请选择SWI影像文件或文件夹')
    first_stage_file_layout.addRow('文件/文件夹:', parent.first_stage_file_path)
    
    first_stage_browse_layout = QHBoxLayout()
    parent.first_stage_browse_button = QPushButton('浏览')
    parent.first_stage_browse_button.clicked.connect(lambda: browse_file_or_folder(parent, 'first_stage'))
    first_stage_browse_layout.addWidget(parent.first_stage_browse_button)
    first_stage_file_layout.addRow('', first_stage_browse_layout)
    first_stage_layout.addLayout(first_stage_file_layout)
    
    # 热力图选项
    first_stage_heatmap_layout = QHBoxLayout()
    parent.first_stage_heatmap_checkbox = QCheckBox('打印热力图')
    parent.first_stage_heatmap_checkbox.setChecked(False)
    first_stage_heatmap_layout.addWidget(parent.first_stage_heatmap_checkbox)
    first_stage_layout.addLayout(first_stage_heatmap_layout)
    
    # 一阶段预测按钮
    parent.first_stage_predict_button = QPushButton('执行一阶段预测')
    parent.first_stage_predict_button.clicked.connect(parent.run_first_stage_prediction)
    first_stage_layout.addWidget(parent.first_stage_predict_button)
    
    # 一阶段进度条
    parent.first_stage_progress_bar = QProgressBar()
    parent.first_stage_progress_bar.setValue(0)
    first_stage_layout.addWidget(parent.first_stage_progress_bar)
    
    # 一阶段预测状态
    parent.first_stage_predict_status = QTextEdit()
    parent.first_stage_predict_status.setReadOnly(True)
    parent.first_stage_predict_status.setPlaceholderText('一阶段预测结果将显示在这里')
    parent.first_stage_predict_status.setMaximumHeight(100)
    first_stage_layout.addWidget(parent.first_stage_predict_status)
    
    # 预测过程数据
    parent.first_stage_prediction_log = QTextEdit()
    parent.first_stage_prediction_log.setReadOnly(True)
    parent.first_stage_prediction_log.setPlaceholderText('预测过程数据将显示在这里')
    parent.first_stage_prediction_log.setMaximumHeight(150)
    first_stage_layout.addWidget(parent.first_stage_prediction_log)
    
    # 二阶段模型预测
    second_stage_group = QGroupBox('二阶段模型预测')
    layout.addWidget(second_stage_group)
    
    second_stage_layout = QVBoxLayout(second_stage_group)
    
    # 二阶段模型选择
    second_stage_model_layout = QFormLayout()
    parent.second_stage_combo = QComboBox()
    parent.second_stage_combo.addItems(['精细分割模型', '边缘优化模型'])
    second_stage_model_layout.addRow('二阶段模型:', parent.second_stage_combo)
    second_stage_layout.addLayout(second_stage_model_layout)
    
    # 预测模式选择
    second_stage_pred_mode_layout = QHBoxLayout()
    
    # 创建按钮组
    parent.second_stage_mode_group = QButtonGroup()
    
    parent.second_stage_single_radio = QRadioButton('单个文件预测')
    parent.second_stage_single_radio.setChecked(True)
    parent.second_stage_mode_group.addButton(parent.second_stage_single_radio)
    second_stage_pred_mode_layout.addWidget(parent.second_stage_single_radio)
    
    parent.second_stage_batch_radio = QRadioButton('批量文件夹预测')
    parent.second_stage_mode_group.addButton(parent.second_stage_batch_radio)
    second_stage_pred_mode_layout.addWidget(parent.second_stage_batch_radio)
    
    # 连接信号
    parent.second_stage_single_radio.clicked.connect(lambda: set_prediction_mode(parent, 'second_stage', 'single'))
    parent.second_stage_batch_radio.clicked.connect(lambda: set_prediction_mode(parent, 'second_stage', 'batch'))
    
    second_stage_layout.addLayout(second_stage_pred_mode_layout)
    
    # 文件选择
    second_stage_file_layout = QFormLayout()
    parent.second_stage_file_path = QLineEdit()
    parent.second_stage_file_path.setPlaceholderText('请选择SWI影像文件或文件夹')
    second_stage_file_layout.addRow('文件/文件夹:', parent.second_stage_file_path)
    
    second_stage_browse_layout = QHBoxLayout()
    parent.second_stage_browse_button = QPushButton('浏览')
    parent.second_stage_browse_button.clicked.connect(lambda: browse_file_or_folder(parent, 'second_stage'))
    second_stage_browse_layout.addWidget(parent.second_stage_browse_button)
    second_stage_file_layout.addRow('', second_stage_browse_layout)
    second_stage_layout.addLayout(second_stage_file_layout)
    
    # 热力图选项
    second_stage_heatmap_layout = QHBoxLayout()
    parent.second_stage_heatmap_checkbox = QCheckBox('打印热力图')
    parent.second_stage_heatmap_checkbox.setChecked(False)
    second_stage_heatmap_layout.addWidget(parent.second_stage_heatmap_checkbox)
    second_stage_layout.addLayout(second_stage_heatmap_layout)
    
    # 二阶段预测按钮
    parent.second_stage_predict_button = QPushButton('执行二阶段预测')
    parent.second_stage_predict_button.clicked.connect(parent.run_second_stage_prediction)
    second_stage_layout.addWidget(parent.second_stage_predict_button)
    
    # 二阶段进度条
    parent.second_stage_progress_bar = QProgressBar()
    parent.second_stage_progress_bar.setValue(0)
    second_stage_layout.addWidget(parent.second_stage_progress_bar)
    
    # 二阶段预测状态
    parent.second_stage_predict_status = QTextEdit()
    parent.second_stage_predict_status.setReadOnly(True)
    parent.second_stage_predict_status.setPlaceholderText('二阶段预测结果将显示在这里')
    parent.second_stage_predict_status.setMaximumHeight(100)
    second_stage_layout.addWidget(parent.second_stage_predict_status)
    
    # 预测过程数据
    parent.second_stage_prediction_log = QTextEdit()
    parent.second_stage_prediction_log.setReadOnly(True)
    parent.second_stage_prediction_log.setPlaceholderText('预测过程数据将显示在这里')
    parent.second_stage_prediction_log.setMaximumHeight(150)
    second_stage_layout.addWidget(parent.second_stage_prediction_log)
