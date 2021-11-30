import random

import pandas as pd

from ChinaVis.QdTree import *

def hashZuobiao(x, y):
    return '%f,%f'%(x, y)

class PollutionCluster:
    def __init__(self):
        self.read_data('./dataset/CN-Reanalysis2016010108.csv')

        self.qdData = QdTree()
        self.polData = {}

        for data in self.timeData:
            self.qdData.Insert(data[-2], data[-1])
            self.polData[hashZuobiao(data[-2], data[-1])] = data[:-2]

        #打印
        # print('后序遍历')
        # print('节点坐标 ','--','划分区域范围')
        # self.qdData.PrintTree()

    def read_data(self, csv_path):
        self.timeData = pd.read_csv(csv_path).to_numpy()[:, :-1]
        self.pointNum = len(self.timeData)

    def cluster(self, k):
        self.centers = [[random.randint(0, self.pointNum-1)] for _ in range(k)]
        for x in self.centers:
            print(self.timeData[x])

    def testFind(self, x):
        x, y = self.timeData[x][-2:]
        self.qdData.Find(x, y)


if __name__ == "__main__":
    pol = PollutionCluster()
    # pol.cluster(10)
    inp = int(input())
    while 0 <= inp < pol.pointNum:
        pol.testFind(random.randint(0, pol.pointNum))
        inp = int(input())