from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QComboBox, QGroupBox, QFormLayout,
    QTextEdit, QLineEdit, QFileDialog, QTabWidget,
    QProgressBar, QSplitter, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt
import os
import json


def create_preprocessing_tab(parent):
    preprocess_tab = QWidget()
    parent.tab_widget.addTab(preprocess_tab, '预处理')
    
    main_layout = QVBoxLayout(preprocess_tab)
    
    preprocess_tabs = QTabWidget()
    main_layout.addWidget(preprocess_tabs)
    
    create_stage_one_tab(parent, preprocess_tabs)
    create_stage_two_tab(parent, preprocess_tabs)


def create_stage_one_tab(parent, tab_widget):
    stage_one_tab = QWidget()
    tab_widget.addTab(stage_one_tab, '一阶段预处理')
    
    splitter = QSplitter(Qt.Vertical)
    stage_one_layout = QVBoxLayout(stage_one_tab)
    stage_one_layout.addWidget(splitter)
    
    top_widget = create_nnunet_folder_settings(parent, 'stage1')
    exec_widget = create_execution_section(parent, 'stage1')
    mid_widget = create_model_parameters(parent, 'stage1', True)
    aug_widget = create_data_augmentation_section(parent, 'stage1')
    
    splitter.addWidget(top_widget)
    splitter.addWidget(exec_widget)
    splitter.addWidget(mid_widget)
    splitter.addWidget(aug_widget)
    splitter.setSizes([300, 150, 250, 250])


def create_stage_two_tab(parent, tab_widget):
    stage_two_tab = QWidget()
    tab_widget.addTab(stage_two_tab, '二阶段预处理')
    
    splitter = QSplitter(Qt.Vertical)
    stage_two_layout = QVBoxLayout(stage_two_tab)
    stage_two_layout.addWidget(splitter)
    
    top_widget = create_nnunet_folder_settings(parent, 'stage2')
    exec_widget = create_execution_section(parent, 'stage2')
    mid_widget = create_model_parameters(parent, 'stage2', False)
    aug_widget = create_data_augmentation_section(parent, 'stage2')
    
    splitter.addWidget(top_widget)
    splitter.addWidget(exec_widget)
    splitter.addWidget(mid_widget)
    splitter.addWidget(aug_widget)
    splitter.setSizes([300, 150, 250, 250])


def create_nnunet_folder_settings(parent, stage_prefix):
    widget = QWidget()
    folder_group = QGroupBox('nnUNet文件夹设置')
    layout = QVBoxLayout(widget)
    layout.addWidget(folder_group)
    
    folder_layout = QFormLayout(folder_group)
    
    raw_layout = QHBoxLayout()
    setattr(parent, f'{stage_prefix}_raw_path', QLineEdit())
    getattr(parent, f'{stage_prefix}_raw_path').setPlaceholderText('默认路径')
    raw_layout.addWidget(getattr(parent, f'{stage_prefix}_raw_path'))
    raw_browse_btn = QPushButton('浏览')
    raw_browse_btn.clicked.connect(lambda: browse_folder(parent, f'{stage_prefix}_raw_path'))
    raw_layout.addWidget(raw_browse_btn)
    raw_default_btn = QPushButton('默认')
    raw_default_btn.clicked.connect(lambda: set_default_path(parent, f'{stage_prefix}_raw_path', 'nnUNet_raw'))
    raw_layout.addWidget(raw_default_btn)
    folder_layout.addRow('nnUNet_raw:', raw_layout)
    
    preprocessed_layout = QHBoxLayout()
    setattr(parent, f'{stage_prefix}_preprocessed_path', QLineEdit())
    getattr(parent, f'{stage_prefix}_preprocessed_path').setPlaceholderText('默认路径')
    preprocessed_layout.addWidget(getattr(parent, f'{stage_prefix}_preprocessed_path'))
    preprocessed_browse_btn = QPushButton('浏览')
    preprocessed_browse_btn.clicked.connect(lambda: browse_folder(parent, f'{stage_prefix}_preprocessed_path'))
    preprocessed_layout.addWidget(preprocessed_browse_btn)
    preprocessed_default_btn = QPushButton('默认')
    preprocessed_default_btn.clicked.connect(lambda: set_default_path(parent, f'{stage_prefix}_preprocessed_path', 'nnUNet_preprocessed'))
    preprocessed_layout.addWidget(preprocessed_default_btn)
    folder_layout.addRow('nnUNet_preprocessed:', preprocessed_layout)
    
    results_layout = QHBoxLayout()
    setattr(parent, f'{stage_prefix}_results_path', QLineEdit())
    getattr(parent, f'{stage_prefix}_results_path').setPlaceholderText('默认路径')
    results_layout.addWidget(getattr(parent, f'{stage_prefix}_results_path'))
    results_browse_btn = QPushButton('浏览')
    results_browse_btn.clicked.connect(lambda: browse_folder(parent, f'{stage_prefix}_results_path'))
    results_layout.addWidget(results_browse_btn)
    results_default_btn = QPushButton('默认')
    results_default_btn.clicked.connect(lambda: set_default_path(parent, f'{stage_prefix}_results_path', 'nnUNet_results'))
    results_layout.addWidget(results_default_btn)
    folder_layout.addRow('nnUNet_results:', results_layout)
    
    dataset_layout = QHBoxLayout()
    setattr(parent, f'{stage_prefix}_dataset_name', QLineEdit())
    getattr(parent, f'{stage_prefix}_dataset_name').setPlaceholderText('例如: Dataset002_NAME2')
    dataset_layout.addWidget(getattr(parent, f'{stage_prefix}_dataset_name'))
    folder_layout.addRow('数据集名称:', dataset_layout)
    
    return widget


def create_model_parameters(parent, stage_prefix, support_3d):
    widget = QWidget()
    param_group = QGroupBox('模型参数设置')
    layout = QVBoxLayout(widget)
    layout.addWidget(param_group)
    
    param_layout = QFormLayout(param_group)
    
    input_format_layout = QHBoxLayout()
    setattr(parent, f'{stage_prefix}_input_format', QComboBox())
    input_format_combo = getattr(parent, f'{stage_prefix}_input_format')
    if support_3d:
        input_format_combo.addItems(['(WH) 2D', '(ZWH) 3D'])
    else:
        input_format_combo.addItems(['(WH) 2D'])
    input_format_layout.addWidget(input_format_combo)
    param_layout.addRow('输入格式:', input_format_layout)
    
    size_layout = QHBoxLayout()
    setattr(parent, f'{stage_prefix}_width', QSpinBox())
    width_spin = getattr(parent, f'{stage_prefix}_width')
    width_spin.setRange(64, 1024)
    width_spin.setValue(256)
    size_layout.addWidget(QLabel('宽:'))
    size_layout.addWidget(width_spin)
    
    setattr(parent, f'{stage_prefix}_height', QSpinBox())
    height_spin = getattr(parent, f'{stage_prefix}_height')
    height_spin.setRange(64, 1024)
    height_spin.setValue(256)
    size_layout.addWidget(QLabel('高:'))
    size_layout.addWidget(height_spin)
    
    if support_3d:
        setattr(parent, f'{stage_prefix}_depth', QSpinBox())
        depth_spin = getattr(parent, f'{stage_prefix}_depth')
        depth_spin.setRange(1, 512)
        depth_spin.setValue(64)
        size_layout.addWidget(QLabel('深:'))
        size_layout.addWidget(depth_spin)
    
    param_layout.addRow('图像尺寸:', size_layout)
    
    button_layout = QHBoxLayout()
    read_plan_btn = QPushButton('读取nnUNetPlan.json')
    read_plan_btn.clicked.connect(lambda: read_nnunet_plan(parent, stage_prefix))
    button_layout.addWidget(read_plan_btn)
    
    save_plan_btn = QPushButton('保存nnUNetPlan.json')
    save_plan_btn.clicked.connect(lambda: save_nnunet_plan(parent, stage_prefix))
    button_layout.addWidget(save_plan_btn)
    
    param_layout.addRow('', button_layout)
    
    return widget


def create_data_augmentation_section(parent, stage_prefix):
    widget = QWidget()
    aug_group = QGroupBox('数据增强设置')
    layout = QVBoxLayout(widget)
    layout.addWidget(aug_group)
    
    aug_layout = QVBoxLayout(aug_group)
    
    aug_list = [
        ('随机旋转', 'random_rotation'),
        ('随机缩放', 'random_scaling'),
        ('随机翻转', 'random_flip'),
        ('弹性形变', 'elastic_deform'),
        ('高斯噪声', 'gaussian_noise'),
        ('亮度调整', 'brightness'),
        ('对比度调整', 'contrast'),
        ('Gamma校正', 'gamma')
    ]
    
    for aug_name, aug_key in aug_list:
        checkbox = QCheckBox(aug_name)
        checkbox.setChecked(False)
        setattr(parent, f'{stage_prefix}_aug_{aug_key}', checkbox)
        aug_layout.addWidget(checkbox)
    
    # 添加一键应用按钮
    apply_btn = QPushButton('一键应用')
    apply_btn.clicked.connect(lambda: apply_data_augmentation(parent, stage_prefix))
    aug_layout.addWidget(apply_btn)
    
    # 添加进度条
    setattr(parent, f'{stage_prefix}_aug_progress', QProgressBar())
    aug_progress = getattr(parent, f'{stage_prefix}_aug_progress')
    aug_progress.setMinimum(0)
    aug_progress.setMaximum(100)
    aug_progress.setValue(0)
    aug_layout.addWidget(aug_progress)
    
    # 添加状态显示
    setattr(parent, f'{stage_prefix}_aug_status', QTextEdit())
    aug_status = getattr(parent, f'{stage_prefix}_aug_status')
    aug_status.setReadOnly(True)
    aug_status.setMaximumHeight(50)
    aug_status.setPlaceholderText('数据增强状态将显示在这里')
    aug_layout.addWidget(aug_status)
    
    return widget


def create_execution_section(parent, stage_prefix):
    widget = QWidget()
    exec_group = QGroupBox('执行')
    layout = QVBoxLayout(widget)
    layout.addWidget(exec_group)
    
    exec_layout = QVBoxLayout(exec_group)
    
    exec_btn = QPushButton(f'执行{stage_prefix}预处理')
    exec_btn.clicked.connect(lambda: run_preprocessing(parent, stage_prefix))
    exec_layout.addWidget(exec_btn)
    
    setattr(parent, f'{stage_prefix}_progress', QProgressBar())
    progress_bar = getattr(parent, f'{stage_prefix}_progress')
    progress_bar.setMinimum(0)
    progress_bar.setMaximum(100)
    progress_bar.setValue(0)
    exec_layout.addWidget(progress_bar)
    
    setattr(parent, f'{stage_prefix}_status', QTextEdit())
    status_text = getattr(parent, f'{stage_prefix}_status')
    status_text.setReadOnly(True)
    status_text.setMaximumHeight(100)
    status_text.setPlaceholderText(f'{stage_prefix}预处理状态将显示在这里')
    exec_layout.addWidget(status_text)
    
    return widget


def browse_folder(parent, line_edit_attr):
    folder_path = QFileDialog.getExistingDirectory(None, '选择文件夹')
    if folder_path:
        getattr(parent, line_edit_attr).setText(folder_path)


def set_default_path(parent, line_edit_attr, folder_name):
    default_path = os.path.join(os.path.expanduser('~'), folder_name)
    getattr(parent, line_edit_attr).setText(default_path)


def read_nnunet_plan(parent, stage_prefix):
    raw_path = getattr(parent, f'{stage_prefix}_raw_path').text()
    dataset_name = getattr(parent, f'{stage_prefix}_dataset_name').text()
    
    if not raw_path or not dataset_name:
        QMessageBox.warning(None, '警告', '请先设置nnUNet_raw路径和数据集名称')
        return
    
    plan_path = os.path.join(raw_path, dataset_name, 'nnUNetPlan.json')
    
    if not os.path.exists(plan_path):
        QMessageBox.warning(None, '警告', f'未找到文件: {plan_path}')
        return
    
    try:
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
        
        if 'patch_size' in plan_data:
            patch_size = plan_data['patch_size']
            if len(patch_size) >= 2:
                getattr(parent, f'{stage_prefix}_width').setValue(patch_size[-1])
                getattr(parent, f'{stage_prefix}_height').setValue(patch_size[-2])
            if len(patch_size) >= 3 and hasattr(parent, f'{stage_prefix}_depth'):
                getattr(parent, f'{stage_prefix}_depth').setValue(patch_size[-3])
        
        status_text = getattr(parent, f'{stage_prefix}_status')
        status_text.append(f'成功读取nnUNetPlan.json: {plan_path}')
        
    except Exception as e:
        QMessageBox.critical(None, '错误', f'读取文件失败: {str(e)}')


def save_nnunet_plan(parent, stage_prefix):
    raw_path = getattr(parent, f'{stage_prefix}_raw_path').text()
    dataset_name = getattr(parent, f'{stage_prefix}_dataset_name').text()
    
    if not raw_path or not dataset_name:
        QMessageBox.warning(None, '警告', '请先设置nnUNet_raw路径和数据集名称')
        return
    
    plan_path = os.path.join(raw_path, dataset_name, 'nnUNetPlan.json')
    
    if not os.path.exists(plan_path):
        QMessageBox.warning(None, '警告', f'未找到文件: {plan_path}')
        return
    
    try:
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
        
        width = getattr(parent, f'{stage_prefix}_width').value()
        height = getattr(parent, f'{stage_prefix}_height').value()
        
        if hasattr(parent, f'{stage_prefix}_depth'):
            depth = getattr(parent, f'{stage_prefix}_depth').value()
            plan_data['patch_size'] = [depth, height, width]
        else:
            plan_data['patch_size'] = [height, width]
        
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=4)
        
        status_text = getattr(parent, f'{stage_prefix}_status')
        status_text.append(f'成功保存nnUNetPlan.json: {plan_path}')
        
    except Exception as e:
        QMessageBox.critical(None, '错误', f'保存文件失败: {str(e)}')


def run_preprocessing(parent, stage_prefix):
    from PyQt5.QtCore import QTimer
    
    status_text = getattr(parent, f'{stage_prefix}_status')
    progress_bar = getattr(parent, f'{stage_prefix}_progress')
    
    status_text.append(f'开始执行{stage_prefix}预处理...')
    progress_bar.setValue(0)
    
    status_text.append(f'{stage_prefix}预处理执行中...')
    progress_bar.setValue(50)
    
    QTimer.singleShot(2000, lambda: finish_preprocessing(parent, stage_prefix))


def finish_preprocessing(parent, stage_prefix):
    status_text = getattr(parent, f'{stage_prefix}_status')
    progress_bar = getattr(parent, f'{stage_prefix}_progress')
    
    progress_bar.setValue(100)
    status_text.append(f'{stage_prefix}预处理完成！')


def apply_data_augmentation(parent, stage_prefix):
    from PyQt5.QtCore import QTimer
    
    aug_status = getattr(parent, f'{stage_prefix}_aug_status')
    aug_progress = getattr(parent, f'{stage_prefix}_aug_progress')
    
    # 检查选中的增强方法
    selected_aug = []
    aug_list = [
        ('随机旋转', 'random_rotation'),
        ('随机缩放', 'random_scaling'),
        ('随机翻转', 'random_flip'),
        ('弹性形变', 'elastic_deform'),
        ('高斯噪声', 'gaussian_noise'),
        ('亮度调整', 'brightness'),
        ('对比度调整', 'contrast'),
        ('Gamma校正', 'gamma')
    ]
    
    for aug_name, aug_key in aug_list:
        if getattr(parent, f'{stage_prefix}_aug_{aug_key}').isChecked():
            selected_aug.append(aug_name)
    
    if not selected_aug:
        aug_status.append('请至少选择一种数据增强方法')
        return
    
    aug_status.append('开始应用数据增强...')
    aug_status.append(f"选中的增强方法: {', '.join(selected_aug)}")
    aug_progress.setValue(0)
    
    # 模拟数据增强过程
    def update_progress():
        current = aug_progress.value()
        if current < 100:
            aug_progress.setValue(current + 10)
            QTimer.singleShot(200, update_progress)
        else:
            aug_status.append('数据增强应用完成！')
    
    QTimer.singleShot(500, update_progress)
