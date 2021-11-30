import pandas as pd
import numpy as np
import random
import math
import time
import itertools

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("output.csv")
sumOfEdge = sum(df.label)

linked_df = df[df.label==1]
row_num = max(linked_df.disease_id) + 1


###初始化点集V
Vertex = np.zeros([row_num,7],dtype=np.int16)

Vertex[:,0] = np.arange(0, row_num)
Vertex[:,1:4] = np.random.randint(2000, size=(row_num, 3))

# ['vid','posX','posY','posZ','forceX','forceY','forceZ']
V = Vertex.tolist()

###初始化边集

E = [[dr.lncRNA_id, dr.disease_id] for i, dr in linked_df.iterrows()]
###3维空间中力引导布局
###定义常量
steps = 200 #迭代步数
K = 0.5 #弹簧劲度系数
k = 0.5 #引力系数
maxOffset = 500
countV = len(V)
countE = len(E)


start_time = time.time()
print(pd.DataFrame(V, columns=['vid','posX','posY','posZ','forceX','forceY','forceZ']).sort_values(by="posX"))
for s in range(steps):
    for A, B in itertools.combinations(V, 2):
            disX=((A[1] - B[1])+0.01)
            disY=((A[2] - B[2])+0.01)
            disZ=((A[3] - B[3])+0.01)
            A[4] += k/math.pow(disX, 2)
            A[5] += k/math.pow(disY, 2)
            A[6] += k/math.pow(disZ, 2)
            B[4] += k/math.pow(-disX + 0.02, 2)
            B[5] += k/math.pow(-disY + 0.02, 2)
            B[6] += k/math.pow(-disZ + 0.02, 2)

    for a,b in E:
        disX = V[a][1] - V[b][1]
        disY = V[a][2] - V[b][2]
        disZ = V[a][3] - V[b][3]
        V[a][4] -= K * disX
        V[b][4] += K * disX
        V[a][5] -= K * disY
        V[b][5] += K * disY
        V[a][6] -= K * disZ
        V[b][6] += K * disZ

    # print(pd.DataFrame(V, columns=['vid', 'posX', 'posY', 'posZ', 'forceX', 'forceY', 'forceZ']).sort_values(by="posX"))

    for v in V:
        v[1] += max(min(maxOffset,0.001*v[4]),-1*maxOffset)
        v[2] += max(min(maxOffset,0.001*v[5]),-1*maxOffset)
        v[3] += max(min(maxOffset,0.001*v[6]),-1*maxOffset)
        v[4],v[5],v[6] = 0, 0, 0

    print("time cost: ",time.time() - start_time)
    start_time = time.time()

    t_V = pd.DataFrame(V, columns=['vid','posX','posY','posZ','forceX','forceY','forceZ'])

    # print(t_V)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(t_V.posX,t_V.posY,t_V.posZ)
    for a, b in E:
        ax.plot([V[a][1], V[b][1]], [V[a][2], V[b][2]], [V[a][3],V[b][3]],linewidth=0.1, alpha=0.5, c='r')
    plt.show()
