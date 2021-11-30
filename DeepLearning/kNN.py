import collections

import numpy as np
from utils import culEuDistance


def knn(X, dataSet, labels, k):
    """
    比较预测数据与历史数据集的欧氏距离，选距离最小的k个历史数据中最多的分类。
    :param X:           需要预测的数据特征
    :param dataSet:     历史数据的数据特征
    :param labels:      与dataSet对应的标签
    :param k:           前k个
    :return:            label标签
    """
    if isinstance(dataSet, list):
        dataSet = np.array(dataSet)
    rowNum = dataSet.shape[0]
    X = np.tile(X,(rowNum,1))
    distances = np.empty(rowNum)
    for row in range(rowNum):
        distances[row] = culEuDistance(X[row], dataSet[row])
    sortedIdx = distances.argsort()
    candidates = []
    for i in range(k):
        candidates.append(labels[sortedIdx[i]])
    return collections.Counter(candidates).most_common(1)[0][0]

if __name__ == "__main__":
    # print(culEuDistance(np.array([3,4]), np.array([2,1])))
    X = [101,20]
    dataSet = [[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]]
    labels = ['爱情片','爱情片','爱情片','动作片','动作片','动作片']
    print(knn(X,dataSet,labels,k=3))
