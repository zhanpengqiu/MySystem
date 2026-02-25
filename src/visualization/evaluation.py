import numpy as np
import matplotlib.pyplot as plt
from src.visualization.image_display import ImageDisplay

class Evaluator:
    """分割评估工具"""
    
    def __init__(self):
        self.image_display = ImageDisplay()
    
    def calculate_dice(self, prediction, ground_truth, threshold=0.5):
        """
        计算Dice系数
        
        Args:
            prediction: 预测分割掩码
            ground_truth: 真实分割掩码
            threshold: 二值化阈值
            
        Returns:
            dice: Dice系数
        """
        # 二值化预测结果
        prediction_binary = (prediction > threshold).astype(np.float32)
        ground_truth_binary = (ground_truth > 0).astype(np.float32)
        
        # 计算交集和并集
        intersection = np.sum(prediction_binary * ground_truth_binary)
        union = np.sum(prediction_binary) + np.sum(ground_truth_binary)
        
        # 计算Dice系数
        if union == 0:
            return 1.0  # 两者都是空，Dice为1
        else:
            dice = 2 * intersection / union
            return dice
    
    def calculate_iou(self, prediction, ground_truth, threshold=0.5):
        """
        计算IoU（交并比）
        
        Args:
            prediction: 预测分割掩码
            ground_truth: 真实分割掩码
            threshold: 二值化阈值
            
        Returns:
            iou: IoU值
        """
        # 二值化预测结果
        prediction_binary = (prediction > threshold).astype(np.float32)
        ground_truth_binary = (ground_truth > 0).astype(np.float32)
        
        # 计算交集和并集
        intersection = np.sum(prediction_binary * ground_truth_binary)
        union = np.sum(np.maximum(prediction_binary, ground_truth_binary))
        
        # 计算IoU
        if union == 0:
            return 1.0  # 两者都是空，IoU为1
        else:
            iou = intersection / union
            return iou
    
    def calculate_sensitivity(self, prediction, ground_truth, threshold=0.5):
        """
        计算敏感性（召回率）
        
        Args:
            prediction: 预测分割掩码
            ground_truth: 真实分割掩码
            threshold: 二值化阈值
            
        Returns:
            sensitivity: 敏感性
        """
        # 二值化预测结果
        prediction_binary = (prediction > threshold).astype(np.float32)
        ground_truth_binary = (ground_truth > 0).astype(np.float32)
        
        # 计算真阳性和假阴性
        true_positive = np.sum(prediction_binary * ground_truth_binary)
        false_negative = np.sum((1 - prediction_binary) * ground_truth_binary)
        
        # 计算敏感性
        if true_positive + false_negative == 0:
            return 1.0  # 没有正样本，敏感性为1
        else:
            sensitivity = true_positive / (true_positive + false_negative)
            return sensitivity
    
    def calculate_specificity(self, prediction, ground_truth, threshold=0.5):
        """
        计算特异性
        
        Args:
            prediction: 预测分割掩码
            ground_truth: 真实分割掩码
            threshold: 二值化阈值
            
        Returns:
            specificity: 特异性
        """
        # 二值化预测结果
        prediction_binary = (prediction > threshold).astype(np.float32)
        ground_truth_binary = (ground_truth > 0).astype(np.float32)
        
        # 计算真阴性和假阳性
        true_negative = np.sum((1 - prediction_binary) * (1 - ground_truth_binary))
        false_positive = np.sum(prediction_binary * (1 - ground_truth_binary))
        
        # 计算特异性
        if true_negative + false_positive == 0:
            return 1.0  # 没有负样本，特异性为1
        else:
            specificity = true_negative / (true_negative + false_positive)
            return specificity
    
    def evaluate(self, prediction, ground_truth, threshold=0.5):
        """
        计算所有评估指标
        
        Args:
            prediction: 预测分割掩码
            ground_truth: 真实分割掩码
            threshold: 二值化阈值
            
        Returns:
            metrics: 评估指标字典
        """
        metrics = {
            'dice': self.calculate_dice(prediction, ground_truth, threshold),
            'iou': self.calculate_iou(prediction, ground_truth, threshold),
            'sensitivity': self.calculate_sensitivity(prediction, ground_truth, threshold),
            'specificity': self.calculate_specificity(prediction, ground_truth, threshold)
        }
        return metrics

class ResultVisualizer:
    """结果可视化工具"""
    
    def __init__(self):
        self.image_display = ImageDisplay()
        self.evaluator = Evaluator()
    
    def visualize_slice_with_mask(self, image_slice, mask_slice, title=""):
        """
        可视化图像切片和分割掩码
        
        Args:
            image_slice: 原始图像切片
            mask_slice: 分割掩码切片
            title: 图像标题
            
        Returns:
            fig: Matplotlib图形对象
        """
        # 归一化图像
        normalized_image = self.image_display.normalize_slice(image_slice)
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示原始图像
        axes[0].imshow(normalized_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 显示分割掩码
        axes[1].imshow(mask_slice, cmap='jet')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # 显示叠加结果
        overlayed = self.image_display.overlay_mask(normalized_image, mask_slice)
        axes[2].imshow(overlayed)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # 设置总标题
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        return fig
    
    def visualize_3d_result(self, image_data, mask_data, slice_indices=None, axis=0):
        """
        可视化3D结果的多个切片
        
        Args:
            image_data: 3D原始图像数据
            mask_data: 3D分割掩码数据
            slice_indices: 要显示的切片索引列表，如果为None则显示中间几个切片
            axis: 切片轴
            
        Returns:
            fig: Matplotlib图形对象
        """
        # 获取图像维度
        if axis == 0:
            depth = image_data.shape[0]
        elif axis == 1:
            depth = image_data.shape[1]
        else:
            depth = image_data.shape[2]
        
        # 如果没有指定切片索引，选择中间的5个切片
        if slice_indices is None:
            middle = depth // 2
            slice_indices = list(range(max(0, middle - 2), min(depth, middle + 3)))
        
        # 计算子图数量
        n_slices = len(slice_indices)
        n_cols = min(3, n_slices)
        n_rows = (n_slices + n_cols - 1) // n_cols
        
        # 创建图形
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.array(axes).reshape(-1)  # 确保axes是一维数组
        
        # 显示每个切片
        for i, slice_idx in enumerate(slice_indices):
            # 获取切片
            image_slice = self.image_display.get_slice(image_data, slice_idx, axis)
            mask_slice = self.image_display.get_slice(mask_data, slice_idx, axis)
            
            # 归一化图像
            normalized_image = self.image_display.normalize_slice(image_slice)
            
            # 叠加掩码
            overlayed = self.image_display.overlay_mask(normalized_image, mask_slice)
            
            # 显示结果
            axes[i].imshow(overlayed)
            axes[i].set_title(f'Slice {slice_idx}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_slices, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics(self, metrics):
        """
        绘制评估指标
        
        Args:
            metrics: 评估指标字典
            
        Returns:
            fig: Matplotlib图形对象
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 准备数据
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # 绘制条形图
        bars = ax.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'purple'])
        
        # 设置标题和标签
        ax.set_title('Segmentation Evaluation Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        
        # 在条形上显示数值
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig