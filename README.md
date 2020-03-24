# Pancreas-Segmentation-using-Deformable-U-Net
## 基于可变卷积模块的U-Net
对传统U-Net模型进行了改进，将可变卷积块融入到U-Net网络中，重新调整网络各层结构，提高传统U-Net对胰腺细胞的切割
## 多方法实现模型评估
对于图像分割，本项目除了使用传统的DSC和Dice指数来评估模型精度，也将图像分割问题转为二分类进行F1值，Precision， Recall的评估，并做出P-R曲线以及ROC曲线和AUC值，多方面全面评估模型性能
