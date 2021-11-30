import matplotlib
import matplotlib.pyplot as plt
import numpy as np
"""
Matplotlib对象简介

   FigureCanvas  画布

   Figure        图

   Axes          坐标轴(实际画图的地方)


"""


def draw_scatter():
    data = np.random.random((100,3))
    for i in range(data.shape[0]):
        data[i][2] = 0 if data[i][2] < 0.5 else 1
    print(data)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(data[:,0], data[:,1], 15)
    ax1 = fig.add_subplot(212)
    # x轴数据， y轴数据， 点大小， 颜色(区间为[x,y])
    ax1.scatter(data[:,0], data[:,1], 15, data[:,2])
    plt.show()

if __name__ == "__main__":
    draw_scatter()