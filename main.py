#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脑微出血分割系统主入口
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication

def main():
    """主函数"""
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = MainWindow()
    
    # 显示主窗口
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()