from model import resnet
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os


# imgs = os.listdir('../ResNet-TensorFlow/test')

true_label = []
pred_label = []

# for i in range(len(imgs)):
#     img_path = os.path.join('..', 'ResNet-TensorFlow', 'test', imgs[i])
#     img = cv2.imread(img_path)
#     img = np.array(img, dtype='float32').reshape((1, 32, 32, 3)) / 255.0
#     out = net(img)
#     pred_label.append(np.argmax(out[0]))
#     true_label.append(int(img_path[-5]))
#     print(len(imgs)-i)
#
# print(pred_label, file=open('pred.txt', 'w'))
# print(true_label, file=open('true.txt', 'w'))


with open('pred.txt', 'r') as file:
    for line in file:
        # print(line)
        line = line[1:-2]
        pred_label = [int(i) for i in line.split(',')]

with open('true.txt', 'r') as file:
    for line in file:
        # print(line)
        line = line[1:-2]
        true_label = [int(i) for i in line.split(',')]


print(accuracy_score(true_label, pred_label))
cm = confusion_matrix(true_label, pred_label)

import matplotlib.pyplot as plt

import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

labels_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
plot_confusion_matrix(cm, labels_name)
plt.show()