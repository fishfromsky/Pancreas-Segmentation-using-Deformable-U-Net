from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
# y_scores = np.array([0.47, 0.75, 0.9, 0.47, 0.55, 0.56, 0.74, 0.62, 0.5, 0.86, 0.8, 0.86, 0.44, 0.67, 0.43, 0.4, 0.52, 0.4, 0.35, 0.1])
# precision, recall, thresholds = precision_recall_curve(y_true, y_scores)


def draw_PR_deform_unet():
    fact = []
    label = []

    file_obj = open('testArrs.txt')
    all_lines = file_obj.readlines()
    for line in all_lines:
        line = line.strip('\n').split(' ')[:-1]
        for data in line:
            fact.append(float(data))

    file_obj.close()

    print('------------读入预测数据完成--------------')

    file_label = open('gt.txt')
    all_label = file_label.readlines()
    for label_sli in all_label:
        label_sli = label_sli.strip('\n').split(' ')[:-1]
        for data in label_sli:
            label.append(int(data))

    file_label.close()

    print('------------读入标记数据完成----------------')

    fact = np.array(fact)
    label = np.array(label)

    precision, recall, threshold = precision_recall_curve(label, fact)

    return precision, recall

def draw_unet_PR():
    fact = []
    label = []

    file_obj = open('testArrs_unet.txt')
    all_lines = file_obj.readlines()
    for line in all_lines:
        line = line.strip('\n').split(' ')[:-1]
        for data in line:
            fact.append(float(data))

    file_obj.close()

    print('------------读入预测数据完成--------------')

    file_label = open('gt_unet.txt')
    all_label = file_label.readlines()
    for label_sli in all_label:
        label_sli = label_sli.strip('\n').split(' ')[:-1]
        for data in label_sli:
            label.append(int(data))

    file_label.close()

    print('------------读入标记数据完成----------------')

    fact = np.array(fact)
    label = np.array(label)

    precision, recall, threshold = precision_recall_curve(label, fact)

    return precision, recall


def draw_PR():
    precision_unet, recall_unet = draw_unet_PR()
    precision_deform, recall_deform = draw_PR_deform_unet()

    plt.figure()
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall_deform, precision_deform, color='red', label='Deformable U-Net')
    plt.plot(recall_unet, precision_unet, label='U-Net')
    plt.legend()
    plt.show()


def draw_ROC_UNet():
    fact = []
    label = []

    file_obj = open('testArrs_unet.txt')
    all_lines = file_obj.readlines()
    for line in all_lines:
        line = line.strip('\n').split(' ')[:-1]
        for data in line:
            fact.append(float(data))

    file_obj.close()

    print('------------读入预测数据完成--------------')

    file_label = open('gt_unet.txt')
    all_label = file_label.readlines()
    for label_sli in all_label:
        label_sli = label_sli.strip('\n').split(' ')[:-1]
        for data in label_sli:
            label.append(int(data))

    file_label.close()

    print('------------读入标记数据完成----------------')

    false_positive_rate, true_positive_rate, threshold = roc_curve(label, fact)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure()
    plt.title('U-Net ROC Curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.plot(false_positive_rate, true_positive_rate, label='AUC=%0.2f'%roc_auc)
    plt.legend()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()


def draw_ROC_Deform():
    fact = []
    label = []

    file_obj = open('testArrs.txt')
    all_lines = file_obj.readlines()
    for line in all_lines:
        line = line.strip('\n').split(' ')[:-1]
        for data in line:
            fact.append(float(data))

    file_obj.close()

    print('------------读入预测数据完成--------------')

    file_label = open('gt.txt')
    all_label = file_label.readlines()
    for label_sli in all_label:
        label_sli = label_sli.strip('\n').split(' ')[:-1]
        for data in label_sli:
            label.append(int(data))

    file_label.close()

    print('------------读入标记数据完成----------------')

    false_positive_rate, true_positive_rate, threshold = roc_curve(label, fact)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure()
    plt.title('Deformable U-Net ROC Curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend()
    plt.plot(false_positive_rate, true_positive_rate, label='AUC=%0.2f' % roc_auc)
    plt.legend()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()


if __name__ == '__main__':
    # draw_PR()
    draw_ROC_UNet()
    # draw_ROC_Deform()