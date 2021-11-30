import math
import os
import numpy as np

def file2matrix(dataPath, dtype=str):
    """
    读取文件中的数据形成 numpy 二维数组
    """
    if not os.path.exists(dataPath):
        print("文件不存在:\t", dataPath)
        return np.array([])
    file = open(dataPath)
    arrOLines = file.readlines()
    reMat = []
    for idx,line in enumerate(arrOLines):
        line = line.strip()
        reMat.append(line.split())
    return np.array(reMat, dtype=dtype)

def culEuDistance(x1, x2):
    """
    计算欧氏距离
    """
    return ((x1 - x2)**2).sum()**0.5

def autoNorm(dataCol):
    """
    数据归一化  (v - min)/(max - min)
    :param dataCol: numpy 数据列
    :return: 归一化的数据列 [0,1]
    """
    minVal = dataCol.min(0)
    maxVal = dataCol.max(0)
    row = dataCol.shape[0]
    normDataCol = (dataCol - np.tile(minVal,(row,1))) / np.tile(maxVal-minVal,(row,1))
    print(dataCol)
    print(np.tile(minVal,(row,1)))
    return normDataCol

def calcShannonEnt(dataCol):
    """
    信息熵
    H = -pi * log2( pi )
    pi 为第 i 个值在所有值中出现的概率
    :param dataSet:
    :return:
    """
    labelNum = dataCol.shape[0]
    labelCounts = {}
    for label in dataCol:
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1
    entroy = 0.0
    for label, count in labelCounts.items():
        # 标签值在所有值中的概率
        prob = count / labelNum
        entroy -= prob * math.log(prob,2)
    return  entroy

if __name__ == "__main__":
    print(">>>>>>>>>>>")
    # print(file2matrix('C:/SameLin/Code/python/Algorithm/Datasets/dataset.txt',int))
    # print(autoNorm(np.array([1,3,12,23,31,0,-12]).T))
    print(calcShannonEnt(np.array([1,3,12,23,31,0,-12]).T))

