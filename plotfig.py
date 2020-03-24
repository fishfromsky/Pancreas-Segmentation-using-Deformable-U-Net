import matplotlib.pylab as plt
import pandas as pd
import pylab as mpl
from utils import *
import cv2


def plot_unet():
    dataset = pd.read_csv('D://UnetDataBase/logs/unet_fd0_Z_ep10_lr1e-05.csv', index_col=0, header=0)
    values = dataset.values
    dice_coef = values[:, 0]
    loss = 1 + values[:, 1]
    return dice_coef, loss


def plot_deform_unet():
    dataset = pd.read_csv('D://UnetDataBase/logs/unet_deform_fd0_Z_ep10_lr0.01.csv', index_col=0, header=0)
    values = dataset.values
    dice_coef = values[:, 0]
    loss = 1 + values[:, 1]
    return dice_coef, loss


def plot():
    plt.figure()
    xlabel = [x for x in range(1, 11)]
    dice_unet, loss_unet = plot_unet()
    dice_deform, loss_deform = plot_deform_unet()
    print(dice_unet)
    print(dice_deform)

    plt.plot(xlabel, dice_unet, label='U-Net', color='cyan')
    plt.plot(xlabel, dice_deform, label='Deformable U-Net', color='orangered')

    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.title('U-Net和Deformable U_Net精度对比')
    plt.legend()
    plt.show()

    plt.figure()
    xlabel = [x for x in range(1, 11)]
    dice_unet, loss_unet = plot_unet()
    dice_deform, loss_deform = plot_deform_unet()

    plt.plot(xlabel, loss_unet, label='U-Net', color='cyan')
    plt.plot(xlabel, loss_deform, label='Deformable U-Net', color='orangered')

    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.title('U-Net和Deformable U_Net损失值对比')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()