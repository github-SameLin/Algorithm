import random
from copy import deepcopy


class GATemple:

    '''
    定义一些遗传算法需要的参数
    '''
    CROSS_RATE = 0.9    # 遗传时的交叉率
    CHANGE_RATE = 0.09   # 遗传时的变异率
    TRANS_COUNT = 5     # 变异时的基因换位次数
    MOUNT_TIMES = 20    # 爬山算法时的迭代次数

    ITER_TIMES = 80     # 迭代次数

    def __init__(self, row_num ):
        '''
        初始化
        '''
        self.row_num = row_num
        pass

    def getGene(self):
        '''
        初始数据生成 —— 染色体
        :return:
        '''
        pass

    def getFitness(self, row):
        pass

    def doChange(self, row):
        '''
        基因变异
        将线路中的两个因子执行交换
        :param row:
        :return:
        '''
        return row

    def change(self, row):
        '''
        染色体变异
        :param row:
        :return:
        '''
        if random.random() < self.CHANGE_RATE:
            for i in range(self.TRANS_COUNT):


                self.doChange(row)
        return row

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

    def run(self):
        print("start")

        '''
            生成初始结果集 —— 染色体集
            计算适应度
        '''

        mat = [ self.getGene() for _ in range(self.row_num) ]
        fits = [self.getFitness(row) for row in mat]

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

            '''
            对最优解使用爬山算法促使其自我进化
            '''
            bestMat = self.clMountain(mat[bestIdx][:])
            bestFit = self.getFitness(bestMat)

            nextMat.append(bestMat)
            nextFits.append(bestFit)

            '''
            开始遗传
            '''
            nextIdx = 1
            while nextIdx < self.row_num:

                if random.random() < self.CROSS_RATE and nextIdx + 1 < self.row_num:
                    '''
                    交叉
                    '''
                    # nextMat.append(a)
                    # nextFits.append(self.getFitness(a))
                    # nextMat.append(b)
                    # nextFits.append(self.getFitness(b))
                else:
                    '''
                    变异
                    '''
                    # a = deepcopy(x)
                    # self.change(a)
                    # nextMat.append(a)
                    # nextFits.append(self.getFitness(a))

            mat, fits = nextMat, nextFits


        '''
        输出结果
        '''

        print("迭代完成")

        bestIdx = fits.index(min(fits))
        bestMat = mat[bestIdx]
        bestFit = fits[bestIdx]

        print("最优权值为:",(bestFit))

        print("最优结果为:", bestMat)

        print("    ")