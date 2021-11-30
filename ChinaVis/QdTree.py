class Point:
    '''构造储存点的类'''

    def __init__(self, x, y):
        self.x = x
        self.y = y


'''区域四叉树（不是平衡四叉树），直观反映空间规律，只有叶节点储存数据'''


class QdNode:
    '''节点类'''

    def __init__(self, x=None, y=None, nRect=[18, 54, 73, 136]):    # 中国地图差不多在这个范围内
        # Rect_范围
        self.pt = Point(x, y) if x else None
        self.block = [None] * 4
        self.nRect = nRect

    def __repr__(self):
        return "pt: {},{}\t nRect:{}\n".format(self.pt.x, self.pt.y, self.nRect)

class QdTree:
    '''树结构'''

    def __init__(self):
        self.root = QdNode()
        self.tRect = self.root.nRect  # 将根节点的范围赋予树
        self.NE = 0
        self.SE = 1
        self.SW = 2
        self.NW = 3

    def comp(self, x, y, node):
        '''定义一个比较函数，以免后期重复编写'''
        r_x = (node.nRect[0] + node.nRect[2]) / 2
        r_y = (node.nRect[1] + node.nRect[3]) / 2  # 节点范围中心坐标
        if x >= r_x and y >= r_y:
            return self.NE
        if x >= r_x and y < r_y:
            return self.SE
        if x < r_x and y <= r_y:
            return self.SW
        if x < r_x and y >= r_y:
            return self.NW


    def ch_r(self, dr, rec):
        '''对节点进行分裂'''
        if dr == self.NE:
            return [(rec[0] + rec[2]) / 2, (rec[1] + rec[3]) / 2, rec[2], rec[3]]
        if dr == self.SE:
            return [(rec[0] + rec[2]) / 2, rec[1], rec[2], (rec[1] + rec[3]) / 2]
        if dr == self.SW:
            return [rec[0], rec[1], (rec[0] + rec[2]) / 2, (rec[1] + rec[3]) / 2]
        if dr == self.NW:
            return [rec[0], (rec[1] + rec[3]) / 2, (rec[0] + rec[2]) / 2, rec[3]]


    def _Add(self, x, y, node):
        if x < self.tRect[0] or x > self.tRect[2] or y < self.tRect[1] or y > self.tRect[3]:  # 超出树所定义的范围
            return

        dr = self.comp(x, y, node)  # 获取方向
        if node.block[dr] == None:
            node.block[dr] = QdNode(x, y, self.ch_r(dr, node.nRect))
        elif node.block[dr].pt == None:
            self._Add(x, y, node.block[dr])
        else:
            var = node.block[dr]  # 用变量var记录原有的节点数据
            node.block[dr] = QdNode(nRect=node.block[dr].nRect)  # 将该节点数据进行Override，只保留nRect数据
            self._Add(var.pt.x, var.pt.y, node.block[dr])
            self._Add(x, y, node.block[dr])

    def _Query(self, x, y, node):
        if x < self.tRect[0] or x > self.tRect[2] or y < self.tRect[1] or y > self.tRect[3]:  # 超出树所定义的范围
            return False
        if not node: return False
        if node.pt:
            return node.pt.x == x and node.pt.y == y
        dr = self.comp(x, y, node)
        return self._Query(x, y, node.block[dr])


    #实现遍历与打印
    def _QTreePost(self, node):
        '''后序遍历'''
        if node == None:
            return

        for dr in range(4):
            self._QTreePost(node.block[dr])

        if node.pt != None:
            print('(%.d,%.d)\t' % (node.pt.x, node.pt.y), '--', node.nRect)  # 存有数据的叶节点

        else:
            print('None\t', '--', node.nRect)  # 只有范围，没有数据的分支节点

    def Insert(self, x, y):
        self._Add(x, y, self.root)

    def PrintTree(self):
        self._QTreePost(self.root)


    def Find(self, x, y):
        print(self._Query(x, y, self.root))


if __name__=='__main__':
    import random
    qd=QdTree()
    #在节点默认范围0~100内生成随机列表
    xli=[random.randint(1,99) for i in range(0,10)]
    yli=[random.randint(1,99) for i in range(0,10)]
    #构造树
    for i in range(0,9):
        qd.Insert(xli[i],yli[i])
    #打印
    print('后序遍历')
    print('节点坐标 ','--','划分区域范围')
    qd.PrintTree()

