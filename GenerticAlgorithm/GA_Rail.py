import bisect
import random
from copy import deepcopy
from math import inf

import matplotlib
import matplotlib.pyplot as plt


class GATemple:
    '''
    定义一些遗传算法需要的参数
    '''
    TRANS_COUNT = 5  # 变异时的基因换位次数
    MOUNT_TIMES = 20  # 爬山算法时的迭代次数

    ITER_TIMES = 2000  # 迭代次数

    def __init__(self, row_num, Pc, Pm):
        '''
        初始化
        '''
        self.row_num = row_num

        self.CROSS_RATE = Pc  # 遗传时的交叉率
        self.CHANGE_RATE = Pm  # 遗传时的变异率

        self.bus_num = 15
        self.rail_num = 2

        self.Vol_max = 45  # 公交车容量 人/车
        self.Vb = 20  # 公交车平均行驶速度 km/h
        self.Vr = 40  # 城市轨道交通平均行驶速度 km/h
        self.Fr = 20  # 城市轨道交通的发车频率 辆/小时
        self.Route_max = 8  # 最大接运线路数量 条
        self.Route_dis_max = 10  # 最大接运线路长度 km
        self.Route_dis_min = 2  # 最短接运线路长度 km
        self.Route_vol_max = 1400

        self.Vol = [[174, 538], [1287, 1345], [229, 129], [177, 159], [171, 293], [1245, 1300], [206, 381], [192, 182],
                    [301, 403], [327, 463], [164, 303], [408, 359], [321, 489], [501, 198], [245, 798]]

        self.Distance = [
            [0.0, 1.186, 1.091, 1.765, 1.832, 1.32, 1.369, 1.198, 0.83, 1.335, 0.685, 1.338, 0.829, 1.451, 0.839, 0.676,
             0.503],
            [1.186, 0.0, 2.254, 2.85, 2.595, 1.953, 0.501, 1.926, 1.828, 2.376, 0.582, 0.911, 0.918, 1.238, 1.348,
             1.116,
             1.591],
            [1.091, 2.254, 0.0, 0.762, 1.353, 1.264, 2.454, 1.096, 0.633, 0.553, 1.772, 2.362, 1.829, 2.356, 1.619,
             1.355,
             0.928],
            [1.765, 2.85, 0.762, 0.0, 1.019, 1.31, 3.122, 1.187, 1.027, 0.487, 2.437, 3.084, 2.556, 3.108, 2.376, 1.806,
             1.688],
            [1.832, 2.595, 1.353, 1.019, 0.0, 0.65, 2.993, 0.675, 1.036, 0.82, 2.37, 3.116, 2.648, 3.281, 2.64, 1.486,
             2.03],
            [1.32, 1.953, 1.264, 1.31, 0.65, 0.0, 2.367, 0.173, 0.696, 0.886, 1.774, 2.53, 2.094, 2.738, 2.159, 0.865,
             1.64],
            [1.369, 0.501, 2.454, 3.122, 2.993, 2.367, 0.0, 2.316, 2.129, 2.67, 0.987, 0.488, 0.763, 0.932, 1.202,
             1.509, 1.65],
            [1.198, 1.926, 1.096, 1.187, 0.675, 0.173, 2.316, 0.0, 0.812, 0.737, 1.697, 2.447, 1.993, 2.633, 2.03,
             0.814,
             1.488],
            [0.83, 1.828, 0.633, 1.027, 1.036, 0.696, 2.129, 0.525, 0.0, 0.545, 1.456, 2.161, 1.66, 2.274, 1.608, 0.812,
             1.0],
            [1.335, 2.376, 0.553, 0.487, 0.82, 0.886, 2.67, 0.737, 0.545, 0.0, 1.99, 2.67, 2.152, 2.74, 2.031, 1.319,
             1.462],
            [0.685, 0.582, 1.772, 2.437, 2.37, 1.744, 0.987, 1.697, 1.456, 1.99, 0.0, 0.757, 0.436, 1.034, 0.804, 0.93,
             1.02],
            [1.338, 0.911, 2.362, 3.084, 3.116, 2.53, 0.488, 2.447, 2.161, 2.67, 0.757, 0.0, 0.533, 0.442, 0.876, 1.686,
             1.469],
            [0.829, 0.918, 1.829, 2.556, 2.648, 2.094, 0.763, 1.993, 1.66, 2.152, 0.436, 0.533, 0.0, 0.646, 0.45, 1.293,
             0.941],
            [1.451, 1.238, 2.356, 3.108, 3.281, 2.738, 0.932, 2.633, 2.274, 2.74, 1.034, 0.442, 0.646, 0.0, 0.743,
             1.935,
             1.426],
            [0.839, 1.348, 1.619, 2.376, 2.64, 2.159, 1.202, 2.03, 1.608, 2.031, 0.804, 0.876, 0.45, 0.743, 0.0, 1.46,
             0.687],
            [0.676, 1.116, 1.355, 1.806, 1.486, 0.865, 1.509, 0.814, 0.525, 1.319, 0.93, 1.686, 1.293, 1.935, 1.46, 0.0,
             1.157],
            [0.503, 1.591, 0.928, 1.688, 2.03, 1.64, 1.65, 1.488, 1.0, 1.462, 1.02, 1.469, 0.941, 1.426, 0.687, 1.157,
             0.0]]

    def getGene(self):
        '''
        初始数据生成 —— 染色体
        '''
        # 先生成bus，然后随机插rail
        raw_row = [i for i in range(self.bus_num)]
        random.shuffle(raw_row)
        row = []
        left = 0
        while left < self.bus_num:
            right = left + random.randint(2, self.bus_num // 2)
            right = min(right, self.bus_num)

            row.extend(raw_row[left: right])
            target = random.randint(0, self.rail_num - 1)

            row.append(self.bus_num + target)
            left = right

        return row

    def getFitness(self, row):
        '''
        计算适应度
        '''
        sum_vol, sum_dis = 0, 0
        target = 0

        route_fail_times = 0
        route_len_fail_times = 0
        route_vol_fail_times = 0

        route, route_dis, route_vol = 0, 0, 0

        tmp_row = [self.bus_num, *row]
        # print(tmp_row[::-1])

        for x in tmp_row[::-1]:
            if x >= self.bus_num:
                target = x

                sum_vol += route_vol
                sum_dis += route_dis

                # 判断约束
                if route > self.Route_max:
                    route += 1
                if route_dis > self.Route_dis_min or route_dis < self.Route_dis_max:
                    route_dis += 1
                if route_vol > self.Route_vol_max:
                    route_vol += 1

                route, route_dis, route_vol = 0, 0, 0

            else:
                # print(x, target)
                route_vol += self.Vol[x][target - self.bus_num]
                route_dis += self.Distance[x][target]
                route += 1

        # 目标函数
        Fk = sum_vol / 40
        # print(Fk, sum_dis, sum_vol)
        res = sum_vol * Fk / 2 + self.Fr * sum_vol / 2 + sum_vol * sum_dis / self.Vb + 1.605 * sum_vol / self.Vr + 5.2 * Fk * sum_dis

        # print("目标函数值",res)

        # 惩罚函数
        t = 0
        if route_fail_times or route_len_fail_times or route_vol_fail_times:
            t = route_fail_times + route_len_fail_times + route_fail_times
        return 1 / (res + t * 1e5)

    def change(self, row):
        '''
        染色体变异
        :param row:
        :return:
        '''
        if random.random() < self.CHANGE_RATE:
            for i in range(self.TRANS_COUNT):
                a = random.randint(0, len(row) - 1)
                b = random.randint(0, len(row) - 1)
                if a == b:
                    continue
                row[a], row[b] = row[b], row[a]

                p = 0
                while p < len(row) - 1:
                    if row[p] >= self.bus_num and row[p + 1] >= self.bus_num:
                        del row[p]
                        continue
                    p += 1

                if row[-1] < self.bus_num:
                    row.append(self.bus_num + random.randint(0, 1))

    def clMountain(self, row):
        '''
        爬山算法
        :param row:
        :return:
        '''
        oldFit = self.getFitness(row)

        for i in range(self.MOUNT_TIMES):

            t_row = self.doChange(row)
            newFit = self.getFitness(row)

            if newFit > oldFit:
                row = t_row
        return row

    def cross(self, fa_row, ma_row):
        '''
        交叉
        :param fa_row:
        :param ma_row:
        :return:
        '''
        row1 = []
        for i in range(len(fa_row)):
            if fa_row[i] >= self.bus_num or fa_row[i] in row1:
                continue
            x = fa_row[i]
            row1.append(x)

            y = 0
            for j in range(i + 1, len(fa_row)):
                if fa_row[j] >= self.bus_num:
                    y = fa_row[j]
                    break

            route, route_dis, route_vol = 1, 0, 0
            while True:
                # print(x)
                pos = 0
                res_dis = inf

                idx = ma_row.index(x)
                nextPos = [fa_row[i + 1], ma_row[idx + 1]]
                if idx > 0:
                    nextPos.append(ma_row[idx - 1])

                for p in nextPos:
                    if self.Distance[x][p] < res_dis and p not in row1 and p < self.bus_num:
                        pos = p
                        res_dis = self.Distance[x][p]

                if res_dis == inf or route > self.Route_max - 1 or route_dis + self.Distance[x][
                    y] > self.Route_dis_max or route_vol > self.Route_vol_max:
                    break

                row1.append(pos)
                route += 1
                route_vol += self.Vol[x][y - self.bus_num]
                if x != fa_row[i]: route_dis += self.Distance[x][pos]

                x = pos
            row1.append(y)
            # print(row1)
        return row1

    def randomSelect(self, ranFit):
        '''
        根据概率随机选择的序列
        :param ranFit:
        :return:
        '''
        ran = random.random()
        return bisect.bisect(ranFit, ran)

    def run(self):
        print("start")

        '''
            生成初始结果集 —— 染色体集
            计算适应度
        '''

        mat = [self.getGene() for _ in range(self.row_num)]
        fits = [self.getFitness(row) for row in mat]

        '''
        绘制图
        '''
        x_data = [x + 1 for x in range(self.ITER_TIMES)]
        y_data = []

        print("开始迭代")
        for time in range(self.ITER_TIMES):

            ''' 
            通过适应度占总适应度的比例生成随机适应度
            '''
            totalFit = sum(fits)
            # 前缀和
            prefixFits = [fits[0] / totalFit]
            for i in range(1, self.row_num):
                prefixFits.append(prefixFits[-1] + fits[i] / totalFit)

            nextMat = []
            nextFits = []

            '''
            上一代中的最优直接遗传到下一代
            '''
            bestIdx = fits.index(min(fits))

            # '''
            # 对最优解使用爬山算法促使其自我进化
            # '''
            # bestMat = self.clMountain(mat[bestIdx][:])
            # bestFit = self.getFitness(bestMat)

            bestRow = mat[bestIdx]
            bestFit = fits[bestIdx]

            nextMat.append(bestRow)
            nextFits.append(bestFit)

            '''
            开始遗传
            '''
            nextIdx = 1
            while nextIdx < self.row_num:
                # 根据概率选取染色体
                fa_idx = self.randomSelect(prefixFits)

                if random.random() < self.CROSS_RATE and nextIdx + 1 < self.row_num:
                    '''
                    交叉
                    '''
                    ma_idx = self.randomSelect(prefixFits)

                    # 规整成线路的二维数组
                    row1, row2 = self.cross(mat[fa_idx], mat[ma_idx]), self.cross(mat[ma_idx], mat[fa_idx])
                    nextMat.append(row1)
                    nextFits.append(self.getFitness(row1))
                    nextMat.append(row2)
                    nextFits.append(self.getFitness(row2))
                    nextIdx += 2
                else:
                    '''
                    变异
                    '''
                    x = mat[fa_idx][:]
                    self.change(x)
                    nextMat.append(x)
                    nextFits.append(self.getFitness(x))
                    nextIdx += 1

            mat, fits = nextMat, nextFits

            '''
            输出结果
            '''

            # print("第 %d 次迭代完成" % (time + 1))
            bestIdx = fits.index(min(fits))
            bestMat = mat[bestIdx]
            bestFit = fits[bestIdx]
            y_data.append(bestFit)

        print("最优权值为:", bestFit)

        print("最优结果为:", [x + 1 for x in bestMat])

        print("    ")

        return [x_data, y_data]


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 5))
    plt.title('遗传算法适应度迭代图')
    plt.xlabel('迭代次数')
    plt.ylabel('适度分数')

    param = [[60, 0.2, 0.3], [60, 0.4, 0.5], [100, 0.4, 0.5], [30, 0.2, 0.4], [20, 0.4, 0.5]]
    for p in param:
        ga = GATemple(*p)
        x_data, y_data = ga.run()
        plt.plot(x_data, y_data, linewidth=2, linestyle='-', )
    labels = [ str(p) for p in param]
    print(labels)
    plt.legend(loc=2, labels=labels)
    plt.show()
