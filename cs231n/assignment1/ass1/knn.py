import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

from cs231n.classifiers import KNearestNeighbor

''' # 加载原始的 CIFAR-10 数据
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# 合理性检查,输出训练集和测试集的大小
print('Training data shape: ', X_train.shape) #(50000, 32, 32, 3)
print('Training labels shape: ', y_train.shape) #(50000,)
print('Test data shape: ', X_test.shape) #(10000, 32, 32, 3)
print('Test labels shape: ', y_test.shape) #(10000,)

# 为了运行效率的提高，取子集运行
# range(5)代表从0到4:[0,1,2,3,4]
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask] # 多维数组取前5000个数组
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# 将多维图像数据重新调整为一行数据，32*32*3->1*3072。-1则表示可以自动推测出
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape) #(5000, 3072) (500, 3072)

# 创建一个 kNN 的分类器实例，训练 kNN 分类器是一个空实现。这个分类器只是简单的记录了一下数据。
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# 测试实现
dists = classifier.compute_distances_two_loops(X_test) '''

