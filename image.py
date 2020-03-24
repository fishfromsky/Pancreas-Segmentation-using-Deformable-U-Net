import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

# data_path = 'D:\\UnetDataBase\\images_Z\\0001\\0133.npy'
# data_path_origine = 'D:\\UnetDataBase\\images\\0001.npy'
# img = np.load(data_path)
# img_origine = np.load(data_path_origine)
# print(img_origine.shape)
# print(img_origine)
# print(img)
# plt.imshow(img)
# plt.set_cmap(plt.gray())
# plt.show()
# plt.title('train_img')
#
# data_path1 = "D:\\UnetDataBase\\labels_Z\\0001\\0133.npy"
# img1 = np.load(data_path1)
# plt.imshow(img1)
# plt.show()
# plt.title("label_img")

data_path = 'D:\\UnetDataBase\\TCIA_pancreas_labels-02-05-2017\\label0001.nii.gz'
img = nib.load(data_path)

width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()

num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.show()









