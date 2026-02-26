from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSpinBox, QComboBox, QTabWidget, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt


def create_image_tab(parent):
    """创建图像加载和展示选项卡"""
    image_tab = QWidget()
    parent.tab_widget.addTab(image_tab, '图像加载')
    
    # 创建布局
    layout = QVBoxLayout(image_tab)
    
    # 创建按钮组
    button_group = QHBoxLayout()
    layout.addLayout(button_group)
    
    # 打开图像文件按钮
    open_image_button = QPushButton('打开图像文件')
    open_image_button.clicked.connect(parent.open_image)
    button_group.addWidget(open_image_button)
    
    # 打开标签文件按钮
    open_label_button = QPushButton('打开标签文件')
    open_label_button.clicked.connect(parent.open_label)
    button_group.addWidget(open_label_button)
    
    # 保存结果按钮
    parent.save_button = QPushButton('保存结果')
    parent.save_button.clicked.connect(parent.save_file)
    button_group.addWidget(parent.save_button)
    
    # 保存所有图像按钮
    parent.save_images_button = QPushButton('保存所有图像')
    parent.save_images_button.clicked.connect(parent.save_images)
    button_group.addWidget(parent.save_images_button)
    
    # 创建保存单个图像按钮组
    parent.save_single_group = QHBoxLayout()
    layout.addLayout(parent.save_single_group)
    
    # 保存图像按钮
    parent.save_image_button = QPushButton('保存图像')
    parent.save_image_button.clicked.connect(lambda: parent.save_single_image('image'))
    parent.save_single_group.addWidget(parent.save_image_button)
    parent.save_image_button.hide()
    
    # 保存标签按钮
    parent.save_label_button = QPushButton('保存标签')
    parent.save_label_button.clicked.connect(lambda: parent.save_single_image('label'))
    parent.save_single_group.addWidget(parent.save_label_button)
    parent.save_label_button.hide()
    
    # 保存融合影像按钮
    parent.save_overlay_button = QPushButton('保存融合影像')
    parent.save_overlay_button.clicked.connect(lambda: parent.save_single_image('overlay'))
    parent.save_single_group.addWidget(parent.save_overlay_button)
    parent.save_overlay_button.hide()
    
    # 创建显示模式选项卡
    parent.display_tab_widget = QTabWidget()
    layout.addWidget(parent.display_tab_widget)
    
    # 创建2D显示选项卡
    create_2d_display_tab(parent)
    
    # 创建ZXY切片展示选项卡
    create_3d_display_tab(parent)
    
    # 创建图像信息区域
    info_group = QGroupBox('图像信息')
    layout.addWidget(info_group)
    
    info_layout = QFormLayout(info_group)
    
    parent.dim_label = QLabel('维度: -')
    info_layout.addRow('维度:', parent.dim_label)
    
    parent.voxel_label = QLabel('体素大小: -')
    info_layout.addRow('体素大小:', parent.voxel_label)
    
    parent.file_label = QLabel('文件: -')
    info_layout.addRow('文件:', parent.file_label)


def create_2d_display_tab(parent):
    """创建2D显示选项卡"""
    tab_2d = QWidget()
    parent.display_tab_widget.addTab(tab_2d, '2D显示')
    
    # 创建布局
    layout = QVBoxLayout(tab_2d)
    
    # 创建图像显示区域
    image_group = QGroupBox('图像展示')
    layout.addWidget(image_group)
    
    image_layout = QVBoxLayout(image_group)
    
    # 创建切片导航控件
    slice_nav_layout = QHBoxLayout()
    
    # 上一切片按钮
    parent.prev_button = QPushButton('上一切片')
    parent.prev_button.clicked.connect(parent.prev_slice)
    parent.prev_button.setEnabled(False)
    slice_nav_layout.addWidget(parent.prev_button)
    
    # 切片信息标签
    parent.slice_label = QLabel('切片: 0/0')
    slice_nav_layout.addWidget(parent.slice_label)
    
    # 切片跳转输入框
    slice_nav_layout.addWidget(QLabel('跳转到:'))
    parent.slice_spin_box = QSpinBox()
    parent.slice_spin_box.setMinimum(1)
    parent.slice_spin_box.setMaximum(1000)
    parent.slice_spin_box.valueChanged.connect(parent.on_slice_jump)
    slice_nav_layout.addWidget(parent.slice_spin_box)
    
    # 切片轴选择
    slice_nav_layout.addWidget(QLabel('切片轴:'))
    parent.slice_axis_combo = QComboBox()
    parent.slice_axis_combo.addItems(['深度轴 (Z)', '高度轴 (Y)', '宽度轴 (X)'])
    parent.slice_axis_combo.currentIndexChanged.connect(parent.on_slice_axis_changed)
    slice_nav_layout.addWidget(parent.slice_axis_combo)
    
    # 下一切片按钮
    parent.next_button = QPushButton('下一切片')
    parent.next_button.clicked.connect(parent.next_slice)
    parent.next_button.setEnabled(False)
    slice_nav_layout.addWidget(parent.next_button)
    
    # 重置按钮
    parent.reset_button = QPushButton('重置')
    parent.reset_button.clicked.connect(parent.reset)
    slice_nav_layout.addWidget(parent.reset_button)
    
    # 将切片导航布局添加到图像布局中
    image_layout.addLayout(slice_nav_layout)
    
    # 创建图像、标签和融合图像显示区域
    parent.image_label_layout = QHBoxLayout()
    image_layout.addLayout(parent.image_label_layout)
    
    # 创建图像显示区域
    parent.image_display_group = QGroupBox('原始图像')
    parent.image_label_layout.addWidget(parent.image_display_group)
    parent.image_label_layout.setStretchFactor(parent.image_display_group, 1)
    
    image_display_layout = QVBoxLayout(parent.image_display_group)
    
    # 创建图像标签
    parent.image_label = QLabel()
    parent.image_label.setAlignment(Qt.AlignCenter)
    parent.image_label.setText('请打开图像文件')
    image_display_layout.addWidget(parent.image_label)
    
    # 创建标签显示区域
    parent.label_display_group = QGroupBox('标签图像')
    parent.image_label_layout.addWidget(parent.label_display_group)
    parent.image_label_layout.setStretchFactor(parent.label_display_group, 1)
    
    label_display_layout = QVBoxLayout(parent.label_display_group)
    
    # 创建标签标签
    parent.label_label = QLabel()
    parent.label_label.setAlignment(Qt.AlignCenter)
    parent.label_label.setText('请打开标签文件')
    label_display_layout.addWidget(parent.label_label)
    
    # 创建融合图像显示区域
    parent.overlay_display_group = QGroupBox('融合图像')
    parent.image_label_layout.addWidget(parent.overlay_display_group)
    parent.image_label_layout.setStretchFactor(parent.overlay_display_group, 1)
    
    overlay_display_layout = QVBoxLayout(parent.overlay_display_group)
    
    # 创建融合图像标签
    parent.overlay_label = QLabel()
    parent.overlay_label.setAlignment(Qt.AlignCenter)
    parent.overlay_label.setText('请打开图像和标签文件')
    overlay_display_layout.addWidget(parent.overlay_label)
    
    # 初始隐藏所有显示区域
    parent.image_display_group.hide()
    parent.label_display_group.hide()
    parent.overlay_display_group.hide()


def create_3d_display_tab(parent):
    """创建ZXY切片展示选项卡"""
    tab_zxy = QWidget()
    parent.display_tab_widget.addTab(tab_zxy, 'ZXY切片展示')
    
    # 创建布局
    layout = QVBoxLayout(tab_zxy)
    
    # 创建ZXY坐标输入组
    input_group = QGroupBox('ZXY坐标输入')
    layout.addWidget(input_group)
    
    input_layout = QHBoxLayout(input_group)
    
    # Z坐标输入
    input_layout.addWidget(QLabel('Z:'))
    parent.z_spin_box = QSpinBox()
    parent.z_spin_box.setMinimum(0)
    parent.z_spin_box.setMaximum(1000)
    input_layout.addWidget(parent.z_spin_box)
    
    # X坐标输入
    input_layout.addWidget(QLabel('X:'))
    parent.x_spin_box = QSpinBox()
    parent.x_spin_box.setMinimum(0)
    parent.x_spin_box.setMaximum(1000)
    input_layout.addWidget(parent.x_spin_box)
    
    # Y坐标输入
    input_layout.addWidget(QLabel('Y:'))
    parent.y_spin_box = QSpinBox()
    parent.y_spin_box.setMinimum(0)
    parent.y_spin_box.setMaximum(1000)
    input_layout.addWidget(parent.y_spin_box)
    
    # 显示按钮
    parent.show_slice_button = QPushButton('显示切片')
    parent.show_slice_button.clicked.connect(parent.show_zxy_slice)
    input_layout.addWidget(parent.show_slice_button)
    
    # 创建切片显示区域
    slice_group = QGroupBox('切片展示')
    layout.addWidget(slice_group)
    
    slice_layout = QHBoxLayout(slice_group)
    
    # Z轴切片显示
    parent.z_slice_group = QGroupBox('Z轴切片')
    slice_layout.addWidget(parent.z_slice_group)
    slice_layout.setStretchFactor(parent.z_slice_group, 1)
    
    z_slice_layout = QVBoxLayout(parent.z_slice_group)
    parent.z_slice_label = QLabel()
    parent.z_slice_label.setAlignment(Qt.AlignCenter)
    parent.z_slice_label.setText('Z轴切片将显示在这里')
    z_slice_layout.addWidget(parent.z_slice_label)
    
    # X轴切片显示
    parent.x_slice_group = QGroupBox('X轴切片')
    slice_layout.addWidget(parent.x_slice_group)
    slice_layout.setStretchFactor(parent.x_slice_group, 1)
    
    x_slice_layout = QVBoxLayout(parent.x_slice_group)
    parent.x_slice_label = QLabel()
    parent.x_slice_label.setAlignment(Qt.AlignCenter)
    parent.x_slice_label.setText('X轴切片将显示在这里')
    x_slice_layout.addWidget(parent.x_slice_label)
    
    # Y轴切片显示
    parent.y_slice_group = QGroupBox('Y轴切片')
    slice_layout.addWidget(parent.y_slice_group)
    slice_layout.setStretchFactor(parent.y_slice_group, 1)
    
    y_slice_layout = QVBoxLayout(parent.y_slice_group)
    parent.y_slice_label = QLabel()
    parent.y_slice_label.setAlignment(Qt.AlignCenter)
    parent.y_slice_label.setText('Y轴切片将显示在这里')
    y_slice_layout.addWidget(parent.y_slice_label)
