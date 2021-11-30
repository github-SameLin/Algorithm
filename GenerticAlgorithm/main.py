# -----------初始数据定义---------------------
# 定义一个17 * 17的二维数组表示站点之间距离 Distance[i][j]
from math import inf
from random import *

Vol = [[174, 538], [1287, 1345], [229, 129], [177, 159], [171, 293], [1245, 1300], [206, 381], [192, 182], [301, 403],
       [327, 463], [164, 303], [408, 359], [321, 489], [501, 198], [245, 798], [206, 381]]

Distance = [
    [0.0, 1.186, 1.091, 1.765, 1.832, 1.32, 1.369, 1.198, 0.83, 1.335, 0.685, 1.338, 0.829, 1.451, 0.839, 0.676, 0.503],
    [1.186, 0.0, 2.254, 2.85, 2.595, 1.953, 0.501, 1.926, 1.828, 2.376, 0.582, 0.911, 0.918, 1.238, 1.348, 1.116,
     1.591],
    [1.091, 2.254, 0.0, 0.762, 1.353, 1.264, 2.454, 1.096, 0.633, 0.553, 1.772, 2.362, 1.829, 2.356, 1.619, 1.355,
     0.928],
    [1.765, 2.85, 0.762, 0.0, 1.019, 1.31, 3.122, 1.187, 1.027, 0.487, 2.437, 3.084, 2.556, 3.108, 2.376, 1.806, 1.688],
    [1.832, 2.595, 1.353, 1.019, 0.0, 0.65, 2.993, 0.675, 1.036, 0.82, 2.37, 3.116, 2.648, 3.281, 2.64, 1.486, 2.03],
    [1.32, 1.953, 1.264, 1.31, 0.65, 0.0, 2.367, 0.173, 0.696, 0.886, 1.774, 2.53, 2.094, 2.738, 2.159, 0.865, 1.64],
    [1.369, 0.501, 2.454, 3.122, 2.993, 2.367, 0.0, 2.316, 2.129, 2.67, 0.987, 0.488, 0.763, 0.932, 1.202, 1.509, 1.65],
    [1.198, 1.926, 1.096, 1.187, 0.675, 0.173, 2.316, 0.0, 0.812, 0.737, 1.697, 2.447, 1.993, 2.633, 2.03, 0.814,
     1.488],
    [0.83, 1.828, 0.633, 1.027, 1.036, 0.696, 2.129, 0.525, 0.0, 0.545, 1.456, 2.161, 1.66, 2.274, 1.608, 0.812, 1.0],
    [1.335, 2.376, 0.553, 0.487, 0.82, 0.886, 2.67, 0.737, 0.545, 0.0, 1.99, 2.67, 2.152, 2.74, 2.031, 1.319, 1.462],
    [0.685, 0.582, 1.772, 2.437, 2.37, 1.744, 0.987, 1.697, 1.456, 1.99, 0.0, 0.757, 0.436, 1.034, 0.804, 0.93, 1.02],
    [1.338, 0.911, 2.362, 3.084, 3.116, 2.53, 0.488, 2.447, 2.161, 2.67, 0.757, 0.0, 0.533, 0.442, 0.876, 1.686, 1.469],
    [0.829, 0.918, 1.829, 2.556, 2.648, 2.094, 0.763, 1.993, 1.66, 2.152, 0.436, 0.533, 0.0, 0.646, 0.45, 1.293, 0.941],
    [1.451, 1.238, 2.356, 3.108, 3.281, 2.738, 0.932, 2.633, 2.274, 2.74, 1.034, 0.442, 0.646, 0.0, 0.743, 1.935,
     1.426],
    [0.839, 1.348, 1.619, 2.376, 2.64, 2.159, 1.202, 2.03, 1.608, 2.031, 0.804, 0.876, 0.45, 0.743, 0.0, 1.46, 0.687],
    [0.676, 1.116, 1.355, 1.806, 1.486, 0.865, 1.509, 0.814, 0.525, 1.319, 0.93, 1.686, 1.293, 1.935, 1.46, 0.0, 1.157],
    [0.503, 1.591, 0.928, 1.688, 2.03, 1.64, 1.65, 1.488, 1.0, 1.462, 1.02, 1.469, 0.941, 1.426, 0.687, 1.157, 0.0]]

Vol_max = 45  # 公交车容量 人/车
Vb = 20  # 公交车平均行驶速度 km/h
Vr = 40  # 城市轨道交通平均行驶速度 km/h
Fr = 20  # 城市轨道交通的发车频率 辆/小时
Route_max = 8  # 最大接运线路数量 条
Route_len_max = 10  # 最大接运线路长度 km
Route_len_min = 2  # 最短接运线路长度 km
Route_vol_max = 1400

# 定义一些遗传算法需要的参数

NP = 50  # 群体规模
JCL = 0.9  # 遗传时的交叉率
BYL = 0.09  # 遗传时的变异率
JYHW = 5  # 变异时的基因换位次数
NG = 800  # 终止代数


def run_GA():
    print("开始迭代")
    # 生成初始结果
    routes = []
    # 获取适应度
    fits = []
    for i in range(NP):
        lines = get_generation()
        routes.append(lines)
        fits.append(fitness(lines))

    ksi = 2
    kc = 0.9

    for i in range(40):
        print("迭代第 %d 次" % i)
        # 求选择概率
        Normfit = [a - 1 + ksi for a in fits]
        sumfit = sum(Normfit)
        Px = [a / sumfit for a in Normfit]

        PPx = [0] * NP
        PPx[0] = Px[0]
        for i in range(1, NP-1):
            PPx[i] = PPx[i-1] + Px[i]
        print("PPx:\t", PPx)

        nxs = []
        # 交叉
        for i in range(NP):
            sita = random()
            for n in range(NP):
                if sita <= PPx[n]:
                    SelFather = n
                    break
            SelMother = randint(0, NP-1)
            r1 = random()
            if r1 <= JCL:
                nx = cross(routes[SelFather], routes[SelMother])
                r2 = random()
                if r2 <= BYL:
                    m1 = [randint(2, 21),randint(2, 21)]
                    nx[m1[0]], nx[m1[1]] = nx[m1[1]], nx[m1[0]]
            nxs.append(nx)
        routes = nx

        for i in range(NP):
            fits[i] = fitness(routes[i])
            ksi = ksi * kc

        ans_fitness = -inf
        ans_route = []
        for i in range(NP):
            if ans_fitness < fitness([i]):
                ans_fitness = fitness([i])
                ans_route = routes[i]

        print("最优结果：")
        print(ans_route)
        print(ans_fitness)


def cross(x, y):
    pass




"""
计算适应度，满足要求，并且目标函数值越小，适应度越高
"""
def fitness(lines):
    res = 0
    for line in lines:
        sum_vol = 0
        sum_dis = 0
        for i in range(len(line) - 1):
            sum_vol += Vol[line[i]][line[-1]-15]
            sum_dis += Distance[line[i]][line[i+1]]
    Fk = sum_vol/40
    res = sum_vol*Fk/2 + Fr*sum_vol/2 + sum_vol*sum_dis/Vb + 1.605*sum_vol/Vr + 5.2*Fk*sum_dis
    return -res


"""
生成符合条件的初始结果
"""
def get_generation():
    M = [a for a in range(15)]
    N = [15, 16]
    routes = []

    while M:
        route_vol = 0
        route_len = 0
        s_idx = N[randint(0, 1)]
        res = [s_idx]
        while M:
            next_idx = randint(0, len(M) - 1)
            route_len += Distance[res[0]][M[next_idx]]
            route_vol += Vol[M[next_idx]][s_idx - 15]
            if len(res) >= Route_max and route_vol > Route_vol_max and route_len > Route_len_max:
                break
            res = [M[next_idx], *res]
            M[next_idx: next_idx + 1] = []
        routes.append(res)
    return routes


if __name__ == "__main__":
    route = get_generation()
    print(route)
    # Dis = "0,1186,1091,1765,1832,1320,1369,1198,830,1335,685,1338,829,1451,839,676,503,1186,0,2254,2850,2595,1953,501,1926,1828,2376,582,911,918,1238,1348,1116,1591,1091,2254,0,762,1353,1264,2454,1096,633,553,1772,2362,1829,2356,1619,1355,928,1765,2850,762,0,1019,1310,3122,1187,1027,487,2437,3084,2556,3108,2376,1806,1688,1832,2595,1353,1019,0,650,2993,675,1036,820,2370,3116,2648,3281,2640,1486,2030,1320,1953,1264,1310,650,0,2367,173,696,886,1774,2530,2094,2738,2159,865,1640,1369,501,2454,3122,2993,2367,0,2316,2129,2670,987,488,763,932,1202,1509,1650,1198,1926,1096,1187,675,173,2316,0,812,737,1697,2447,1993,2633,2030,814,1488,830,1828,633,1027,1036,696,2129,525,0,545,1456,2161,1660,2274,1608,812,1000,1335,2376,553,487,820,886,2670,737,545,0,1990,2670,2152,2740,2031,1319,1462,685,582,1772,2437,2370,1744,987,1697,1456,1990,0,757,436,1034,804,930,1020,1338,911,2362,3084,3116,2530,488,2447,2161,2670,757,0,533,442,876,1686,1469,829,918,1829,2556,2648,2094,763,1993,1660,2152,436,533,0,646,450,1293,941,1451,1238,2356,3108,3281,2738,932,2633,2274,2740,1034,442,646,0,743,1935,1426,839,1348,1619,2376,2640,2159,1202,2030,1608,2031,804,876,450,743,0,1460,687,676,1116,1355,1806,1486,865,1509,814,525,1319,930,1686,1293,1935,1460,0,1157,503,1591,928,1688,2030,1640,1650,1488,1000,1462,1020,1469,941,1426,687,1157,0"
    # Dis = Dis.split(",")
    # print([[int(Dis[j]) / 1000 for j in range(i * 17, (i + 1) * 17)] for i in range(17)])
