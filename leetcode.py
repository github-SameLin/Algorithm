from collections import abc
import bisect
import copy
import functools
import math
import numbers
import operator
from collections import deque
from itertools import permutations
from math import inf
from typing import List
import numpy as np

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = None
        self.right = None

class Solution:
    '''
    67. Add Binary
    '''
    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a,2)+ int(b,2))[2:]
    '''
    268. 缺失的整数  找到最小的没出现的正整数
        原地哈希，把数值规定在 1~n 之间，用数组的元素位置的值的正负表示值是否出现过
    '''
    def missingNumber(self, nums: List[int]) -> int:
        if 1 not in nums:
            return 1

        n = len(nums)
        for i in range(n):
            if nums[i] > n or nums[i] < 1:
                nums[i] = 1

        for num in nums:
            x = abs(num)
            if x == n:
                nums[0] = -abs(nums[0])
            else:
                nums[x] = -abs(nums[x])

        for i in range(1, n):
            if nums[i] > 0:
                return i

        return n if nums[0] > 0 else n+1
    '''
    16. 3Sum Closest 找到数组中三个数的和离制定目标值最近
        排序
        双指针
    '''
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        ans = inf
        for i in range(n - 2):
            if i and nums[i] == nums[i-1]:
                continue
            l,r = i+1, n-1
            while l < r:
                res = nums[l] + nums[r] + nums[i]
                if  res > target:
                    r -= 1
                elif res < target:
                    l += 1
                else:
                    return res
                if abs(res - target) < abs(ans - target):
                    ans = res
        return ans
    '''
    139. Word Break 判断一个句子是否能由词典中的单词拼成
        字符串匹配
        动态规划
    '''
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordDict = list(set(wordDict))
        s_len = len(s)
        dp = [False for _ in range(s_len+1)]
        dp[0] = True
        for pos in range(s_len):
            if dp[pos] == False: continue
            for word in wordDict:
                w_len = len(word)
                if pos + w_len <= s_len and s[pos: pos+w_len] == word:
                    dp[pos+ w_len] = True
        return dp[s_len]
    '''
    5.Longest Palindromic Substring 求字符串中最长的回文子串
    
    '''
    def longestPalindrome(self, s: str) -> str:
        s_len = len(s)
        for l in range(s_len, 0, -1):
            for p in range(s_len-l+1):
                t = s[p:p+l]
                if t == t[::-1]:
                    return s[p:p+l]
        return ""
    '''
    209. Minimum Size Subarray Sum
        前缀和
        二分查找
    '''
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:

        #二分
        def find_erfen(nums, x) -> int:
            l,r = 0, len(nums)-1
            while l < r:
                mid = (l + r) >> 1
                if nums[mid] < x:
                    l = mid + 1
                else:
                    r = mid
            return l if nums[l] >= r else -1
        pre = 0
        ans = inf
        pre_sum = [0]
        for x in nums:
            pre = pre + x
            pre_sum.append(pre)
        for i in range(len(nums)):
            t = s + pre_sum[i]
            x =  find_erfen(pre_sum, t)
            if x != len(pre_sum):
                ans = min(ans, x - i)
        return ans
    '''
    718.最长重复子数组
    '''
    def findLength(self, A: List[int], B: List[int]) -> int:
        a_len, b_len = len(A), len(B)
        dp = [[0]*a_len for _ in range(b_len)]
        ans = 0
        for i in range(a_len):
            for j in range(b_len):
                if A[i] == B[j]:
                    if i and j:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = 1
                    ans = max(ans,dp[i][j])
        return ans

    '''
    63.不同路径 带路障基础题
        动态规划
    '''
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        n = len(obstacleGrid)
        if not n: return 0
        m = len(obstacleGrid[0])
        if not m: return 0
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1,n+1):
            for j in range(1,m+1):
                if not obstacleGrid[i-1][j-1]:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
                    if i==1 and j==1:dp[i][j] = 1
                print("i={},j={},__ {}".format(i,j,dp[i][j]));
        return dp[n][m];
    '''
    108. 将有序数组转换为二叉搜索树
        递归
        平衡二叉搜索树
    '''
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def initTree(l, r):
            if l > r: return
            mid = (l+r+1)>>1
            tmpTree = TreeNode(nums[mid])
            tmpTree.left = initTree(l,mid-1)
            tmpTree.right = initTree(mid+1,r)
            return tmpTree
        return initTree(0,len(nums)-1)
    '''
    112. 路径总和
        搜索
    '''
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if root is None: return False
        res = sum - root.val
        if not root.left and not root.right and res == 0:  return True
        if root.left and self.hasPathSum(root.left, res):   return True
        if root.right and self.hasPathSum(root.right, res): return True
        return False
    '''
    面试题 17.13. 恢复空格
        字符串匹配
        动态规划
    '''
    def respace(self, dictionary: List[str], sentence: str) -> int:
        word_len = [ len(x) for x in dictionary]
        word_max_len = max(word_len)

        s_len = len(sentence)
        dp = [0 for _ in range(s_len+1)]
        for i in range(1, s_len+1):
            dp[i] = dp[i-1] + 1
            for j in range(max(0,i- word_max_len), i):
                if sentence[j : i] in dictionary:
                    dp[i] = min(dp[i], dp[j])
        return dp[s_len]
    '''
    11. 盛最多水的容器 求直方图中面积最大的不连续矩形
        双指针
    '''
    def maxArea(self, height: List[int]) -> int:
        l,r = 0, len(height)-1
        ans = 0
        while l<r:
            if height[l] < height[r]:
                ans = max(ans, height[l] * (r-l))
                l = l + 1
            else:
                ans = max(ans, height[r] * (r-l))
                r = r -1
        return ans

    '''
    6.Z字形变换
        字符排版
    '''
    def convert(self, s: str, numRows: int) -> str:
        ans = ""
        if s == ans: return ans
        if numRows == 1: return s
        stride = numRows* 2 - 2
        step = stride
        s_len = len(s)
        for row in range(numRows):
            for col in range(0,s_len,stride):
                pos = row + col
                if pos >= s_len: break
                if step in [0, stride]:
                    ans += s[pos]
                else:
                    ans += s[pos]
                    if pos + step >= s_len: continue
                    ans += s[pos+step]
            step -= 2
        return ans
    '''
    315. 计算右侧小于当前元素的个数
        归并排序
    '''
    def countSmaller(self, nums: List[int]) -> List[int]:

        def init(self, nums, left, right, temp, indexes, res):
            if left == right:
                return
            mid = (right + left) >> 1

            init(nums, left, mid, temp, indexes, res)
            init(nums, mid + 1, right, temp, indexes, res)

            if nums[indexes[mid]] <= nums[indexes[mid + 1]]:
                return

            sort_and_count_smaller(nums, left, mid, right, temp, indexes, res)

        def sort_and_count_smaller( nums, left, mid, right, temp, indexes, res):
            for i in range(left, right + 1):
                temp[i] = indexes[i]

            l = left
            r = mid + 1
            for i in range(left, right + 1):
                if l > mid:
                    indexes[i] = temp[r]
                    r += 1
                elif r > right:
                    indexes[i] = temp[l]
                    l += 1
                    res[indexes[i]] += (right - mid)
                elif nums[temp[l]] <= nums[temp[r]]:
                    indexes[i] = temp[l]
                    l += 1
                    res[indexes[i]] += (r - mid - 1)
                else:
                    assert nums[temp[l]] > nums[temp[r]]
                    indexes[i] = temp[r]
                    r += 1

        size = len(nums)
        if size == 0:
            return []
        if size == 1:
            return [0]

        temp = [None for _ in range(size)]
        indexes = [i for i in range(size)]
        res = [0 for _ in range(size)]

        init(nums, 0, size-1, temp, indexes, res)
        return res

    '''
    174. 地下城游戏
        搜索
    '''

    class queNode:
        def __init__(self,x,y,val,min_val):
            self.x = x
            self.y = y
            self.val = val
            self.min_val = min_val


    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        transfer = [[1,0],[0,1]]
        n = len(dungeon)
        if not n: return 0
        m,ans = len(dungeon[0]), -inf
        if n == m == 1:
            ans = dungeon[0][0]
        else:
            que = []
            que.append(self.queNode(0,0,dungeon[0][0],dungeon[0][0]))
            while len(que):
                node = que[0]
                que.remove(node)
                for trans in transfer:
                    x,y = node.x + trans[0], node.y + trans[1]
                    if x >= n or y >= m: continue
                    val = node.val + dungeon[x][y]
                    min_val = min(node.min_val, val)
                    if x == n-1 and y == m-1:
                        ans = max(ans, min_val)
                    else:
                        que.append(self.queNode(x, y, val, min_val))
        return 1 if ans > 0 else -ans +1
    '''
    96. 不同的二叉搜索树
        动态规划
    '''
    def numTrees(self, n: int) -> int:
        dp = [0] * (n+1)
        dp[0] = dp[1] = 1
        for i in range(2, n+1):
            dp[i] = dp[i-1]*(4*i-2)//(i+1)

        return dp[n]
# -----------------------------------------------
#         dp = [0] * (n+1)
#         dp[0] = dp[1] = 1
#         for i in range(2, n+1):
#             for j in range(1, i+1):
#                 dp[i] += dp[j-1] * dp[i-j]
#         return dp[n]
    '''
    785. 判断二分图
        并查集
        DFS
    '''
    def isBipartite(self, graph: List[List[int]]) -> bool:

#         #---------------并查集--------------
#         # 数组初始化
#         def init_union(length):
#             return [i for i in range(length)]
#         #查
#         def find(i):
#             return i if roots[i] == i else find(roots[i])
#
#         #判断p 和 q 是否在同一个集合中
#         def is_Union(p, q):
#             return find(p) == find(q)
#
#         #将 p 和 q 添加到同一个集合中
#         def update_union(p, q):
#             roots[find(p)] = find(q)
#
#         #------------全局参数设置------------
#         size = len(graph)
#         roots = init_union(size)
#         #----------------END---------------
#
#         for node in range(size):
#             for anti_node in graph[node]:
#                 if is_Union(node, anti_node):
#                     return False
#                 # 将邻接表里每一个与当前结点链接的点进行集合
#                 update_union(graph[node][0], anti_node)
#         return True

        #-------DFS 图染色-------
        def dfsColor(node, color):
            if visited[node] !=0:
                return visited[node] == color
            visited[node] = color
            for anti_node in graph[node]:
                if not dfsColor(anti_node,-color):
                    return False
            return True



        size = len(graph)
        visited = [0] * size
        for i in range(size):
            if visited[i] == 0 and not dfsColor(i,1):
                return False
            return True

    '''
    35. 搜索插入位置
        二分查找
    '''
    def searchInsert(self, nums: List[int], target: int) -> int:

        # ----------二分查找-----------
        # 找到nums中第一个 大于等于 target的目标位置
        # pos =  bisect.bisct_left(nums,target,left,right)

        # 将target 插入到该位置（搜索是O(log n), 插入却是O(n)）
        # bisect.insort_left(nums,target,left, right)

        # 找到nums中第一个   大于  target的目标位置
        # pos = bisect.bisct(nums,target,left,right)
        # pos = bisect.bisct_right(nums,target,left,right)

        # 将target 插入到该位置（搜索是O(log n), 插入却是O(n)）
        # bisect.insort(nums,target,left, right)
        # bisect.insort_right(nums,target,left, right)

        return bisect.insort_left(nums, target)

    '''
    312. 戳气球
        动态规划
    '''
    def maxCoins(self, nums: List[int]) -> int:
        nums_len = len(nums)
        if nums_len == 0:
            return 0
        nums = [1] + nums + [1]
        size = len(nums)
        # dp[i][j] 代表 区间[i,j]的最大值
        # dp = [[0]*size] * size
        dp = [[0]*size for _ in range(size)]
        # 从长度遍历，先遍历小区间，再到大区间
        for l in range(2, size):
            for i in range(0, size - l):
                j = i+l
                for k in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], nums[i] * nums[k] * nums[j] + dp[i][k] + dp[k][j])
        # for i in range(len(dp)):
        #     print(i,end=":: ")
        #     for j in range(len(dp[i])):
        #         print("{}_{}".format(dp[i][j],id(dp[i][j])),end="\t")
        #     print()
        # print(dp)
        return dp[0][size-1]
    '''
    95. 不同的二叉搜索树 II
        DFS
    '''
    def generateTrees(self, n: int) -> List[TreeNode]:

# --------不同的二叉搜索树 I------------------
#         dp = [0] * (n+1)
#         dp[0] = dp[1] = 1
#         for i in range(2, n+1):
#             for j in range(1, i+1):
#                 dp[i] += dp[j-1] * dp[i-j]
#         return dp[n]
        def insertBalanceTree(left, right) -> List[TreeNode]:
            if left > right:
                return [None]
            res = []
            for i in range(left, right+1):
                left_trees = insertBalanceTree(left, i -1)
                right_trees = insertBalanceTree(i+1, right)
                for left_tree in left_trees:
                    for right_tree in right_trees:
                        tree = TreeNode(i)
                        tree.left = left_tree
                        tree.right = right_tree
                        res.append(tree)
            return res
        if n == 0:
            return []
        return insertBalanceTree(1, n)

    '''
    42. 接雨水
        双指针
    '''
    def trap(self, height: List[int]) -> int:
        size = len(height)
        # 确定池子左墙，找大于等于左墙的右墙，并记录中间bar的累计和
        ans = left = block = 0
        for right in range(1, size):
            if height[right] >= height[left]:
                ans += (right - left - 1) * height[left] - block
                left, block = right, 0
            else:
                block += height[right]
        # 确定池子右墙，找大于右墙的左墙，并记录中间bar的累计和
        right, block = size - 1, 0
        for left in range(size - 2, -1, -1):
            if height[left] > height[right]:
                ans += (right - left - 1) * height[right] - block
                right,block = left, 0
            else:
                block += height[left]
        return ans

    '''
    592. 分数加减运算
    '''
    def fractionAddition(self, expression: str) -> str:
        def gcd(x, y):
            return x if y == 0 else gcd(y, x % y)

        def simple(x, y):
            flag = 1
            if x < 0:
                flag *= -1
                x *= -1
            if y < 0:
                flag *= -1
                y *= -1
            sub = gcd(x,y)
            if sub != 1:
                x /= sub
                y /= sub
            return x* flag, y
        a, b, c, d = 0, 1, 0, 0
        flag, note = 1, 0
        for i in range(len(expression)):
            if expression[i] in ["-","+"]:
                if c or d:
                    a = a * d
                    c = c * b
                    b = b * d
                    if flag == 1:
                        a += c
                    else:
                        a -= c
                    a, b = simple(a, b)
                flag =  1 if expression[i] == "+" else -1
                note, c, d = 0, 0, 0
            elif expression[i] == "/":
                note = 1
            elif note == 0:
                c = c * 10 + int(expression[i])
            elif note == 1:
                d = d * 10 + int(expression[i])

        if c or d:
            a = a * d
            c = c * b
            b = b * d
            if flag == 1:
                a += c
            else:
                a -= c
            a, b = simple(a, b)
        return str(int(a)) + "/" + str(int(b))

    '''
    1025. 除数博弈
        动态规划
    '''
    def divisorGame(self, N: int) -> bool:
        dp = [False] * 1005
        dp[0],dp[1],dp[2] = True, False, True
        for i in range(3, N+1):
            x = int(math.sqrt(i))
            for j in range(1, x+1):
                if i % j == 0 and dp[i-j] == False:
                    dp[i] = True
                    break
        for i in range(1, N + 1):
            print("{}\t{}".format(i,dp[i]))
        return dp[N]
    '''
    410. 分割数组的最大值
        动态规划
        二分查找
    '''
    def splitArray(self, nums: List[int], m: int) -> int:
        #————————二分查找——————————————————————
        # 边界为[max(nums),sum(nums)]
        # 用二分法在里面找到一个值，使得满足条件
        l, r =  max(nums),sum(nums)
        while l <= r:
            mid, cnt, tmp = (l + r)>> 1, 0, inf
            for num in nums:
                if tmp + num > mid:
                    cnt, tmp = cnt+1, num
                else:
                    tmp += num
            # 如果分割的数量大于m，则这个值偏小
            if cnt > m:
                l = mid + 1
            # 如果分割的数量大于m，则这个值偏大
            # 如果分割的数量等于m，则这个值也可能偏大，不是最小的符合要求的值
            else:
                r = mid - 1
        return l


        #————————动态规划——————————————————————
        # if m == 1:
        #     return sum(nums)
        # size = len(nums)
        # # 前缀和
        # p_nums = [0] * size
        # p_nums[0] = nums[0]
        # for i in range(1, size):
        #     p_nums[i] = p_nums[i-1] + nums[i]
        # # 动态规划
        # dp = [[inf]*(m+1) for _ in range(size+1)]
        # for i in range(1, size+1):
        #     dp[i][1] = p_nums[i-1]
        # for i in range(2, size+1):
        #     for j in range(2, min(m, i)+1):
        #         for k in range(j-1, i):
        #             dp[i][j] = min(dp[i][j], max(dp[k][j-1],p_nums[i-1]-p_nums[k-1]))
        # return int(dp[size][m])

    '''
    343. 整数拆分
    动态规划 
    '''
    def integerBreak(self, n: int) -> int:
        dp = [0] * 60
        dp[1] = dp[2] = 1
        for x in range(3, n+1):
            for i in range(2, round(x/2) + 1):
                dp[x] = max(dp[x], max(i * (x-i), max(dp[i] * dp[x-i], max(i * dp[x-i], dp[i] * (x-i)))))
        return dp[n]

    '''
    336. 回文对
    字符串匹配
    '''
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        ans = []
        size = len(words)
        def findPalind(word, idx, forward):
            # print(word)
            word_len = len(word)
            for i in range(1, word_len):
                mid, offset = divmod(i,2)
                tmp_word = word[:i-1:-1]
                if not forward:
                    tmp_word = tmp_word[::-1]
                if (i == 1 or word[:mid] == word[mid+offset:i]) and tmp_word in words:
                    another = words.index(tmp_word)
                    ans.append([another, idx] if forward else [idx, another])
        for idx,word in enumerate(words):
            if word == "": continue
            if word[::-1] in words[idx+1:]:
                another = words.index(word[::-1], idx+1)
                ans.append([idx,another])
                ans.append([another,idx])
            mid,offset = divmod(len(word),2)
            if "" in words and ((offset and word[:mid] == word[:mid:-1]) or word[:mid] == word[:mid - 1:-1]):
                print(word)
                another = words.index("")
                ans.append([idx,another])
                ans.append([another,idx])
            findPalind(word,idx, True)
            findPalind(word[::-1],idx, False)
        return ans

    '''
    207. 课程表
    判环
    '''
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        pred = [False] * numCourses
        pre = [[] for _ in range(numCourses)]
        for a,b in prerequisites:
            pre[b].append(a)
            pred[a] = True
        num = [0] * numCourses
        for i in range(numCourses):
            if not pred[i]:
                visited = [False] * numCourses
                x = i
                que = deque()
                que.append(x)
                while len(que):
                    x = que.popleft()
                    visited[x] = True
                    num[x] = 1
                    for x_pres in pre[x]:
                        if visited[x_pres]: return False
                        que.append(x_pres)
        return True if sum(num) == numCourses else False

    '''
    529.扫雷游戏
    递归 判断
    '''
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        x,y = click
        size_x, size_y = len(board), len(board[0])
        if board[x][y] == "M":
            board[x][y] = "X"
            return board
        elif board[x][y] == "E":
            que = deque()
            turns = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]
            visited = [[False]*size_y for _ in range(size_x)]
            que.append([x, y])
            while len(que):
                x, y = que.popleft()
                boom_num = 0
                for a,b in turns:
                    xx, yy = x+a, y+b
                    if 0 <= xx < size_x and 0 <= yy < size_y and board[xx][yy] in ["M","X"]:
                            boom_num += 1
                if boom_num:
                    board[x][y] = str(boom_num)
                else:
                    board[x][y] = "B"
                    for a,b in turns:
                        xx, yy = x+a, y+b
                        if 0 <= xx < size_x and 0 <= yy < size_y and not visited[xx][yy]:
                            que.append([xx,yy])
                            visited[xx][yy] = True
            return board

    '''
    647. 回文子串
    '''
    def countSubstrings(self, s: str) -> int:
        def splitString(left, right):
            if s[left] == s[right]:
                # ans.append(s[left:right+1])
                return True
            return False

        # ans = []
        size = len(s)
        res = size
        for i in range(size):
            # ans.append(s[i])

            left, right = i-1, i+1
            while left >= 0 and right < size and splitString(left, right):
                res += 1
                left, right = left-1, right+1

            left, right = i, i+1
            while left >= 0 and right < size and splitString(left, right):
                res += 1
                left, right = left-1, right+1


        return res

    '''
    679. 24 点游戏
    算数 递归
    '''
    def judgePoint24(self, nums: List[int]) -> bool:
        #考虑精度误差，当结果与 24 的误差在 1e6 以内可以认为相等
        EPSILON = 1e-6
        def dfs(numbers, target):
            if len(numbers) == 1:
                return True if abs(numbers[0] - target) < EPSILON else False
            for idx,num in enumerate(numbers):
                tmp_nums = numbers[:]
                tmp_nums.remove(num)
                if dfs(tmp_nums, target - num) or dfs(tmp_nums, num - target) or (target and dfs(tmp_nums, num / target)) or (num and dfs(tmp_nums, target / num)):
                    return True
            return False
        if dfs(nums, 24):
            return True

        # 将 (a b) (c d) 的特殊情况单独处理，其他情况都可以从一个数逐步拆解
        for num_arrange in list(permutations(nums)):
            if (num_arrange[0] + num_arrange[1]) * (num_arrange[2] + num_arrange[3]) == 24 \
                or (num_arrange[0] + num_arrange[1]) * (num_arrange[2] - num_arrange[3])  == 24 \
                or (num_arrange[0] - num_arrange[1]) * (num_arrange[2] + num_arrange[3])  == 24 \
                or (num_arrange[0] - num_arrange[1]) * (num_arrange[2] - num_arrange[3])  == 24 \
                or ((num_arrange[2] + num_arrange[3]) and (num_arrange[0] + num_arrange[1]) / (num_arrange[2] + num_arrange[3])  == 24) \
                or ((num_arrange[2] - num_arrange[3]) and (num_arrange[0] + num_arrange[1]) / (num_arrange[2] - num_arrange[3])  == 24) \
                or ((num_arrange[2] + num_arrange[3]) and (num_arrange[0] - num_arrange[1]) / (num_arrange[2] + num_arrange[3])  == 24) \
                or ((num_arrange[2] - num_arrange[3]) and (num_arrange[0] - num_arrange[1]) / (num_arrange[2] - num_arrange[3])  == 24):
                return True
        return False
        #——————————————大神的暴力美学————————————————
        # def helper(nums):
        #     print(nums)
        #     if len(nums) == 1:
        #         return math.isclose(nums[0], 24)
        #     return any(helper((x,) + tuple(rest)) for a, b, *rest in permutations(nums) for x in
        #                {a + b, a - b, a * b, b and a / b})
        #
        # return helper(tuple(nums))
    '''
    201. 数字范围按位与
    位运算
    '''
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        sub = n-m
        while m:
            x = m&-m
            if sub >= x:
                m -= x
                sub += x
            else:
                break
        return m

    '''
    491. 递增子序列
    地推
    '''
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        ans = [[] for _ in range(201)]
        for idx, num in enumerate(nums):
            ans_ls = ans[num + 100][:]
            for res_ls in ans[:num+101]:
                for res in res_ls:
                    # print(res)
                    tmp_res = res[:]
                    tmp_res.append(num)
                    if tmp_res not in ans[num+100]:
                        ans_ls.append(tmp_res)
            if [num] not in ans_ls:
                ans_ls.append([num])
            ans[num + 100] = ans_ls
        ans = [res for res_ls in ans for res in res_ls if len(res)>1]
        return ans

    '''
    17. 电话号码的字母组合
    递归
    '''
    def letterCombinations(self, digits: str) -> List[str]:
        keyboard = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        ans = []
        if digits=="" :
            return ans
        def letterC(dig, res):
            if len(dig) == 1:
                for letter in keyboard[dig]:
                    ans.append(res + letter)
            else:
                for letter in keyboard[dig[0]]:
                    letterC(dig[1:],res+letter)
        letterC(digits, "")
        return ans

    '''
    915.分割数组
    双向遍历
    '''
    def partitionDisjoint(self, A: List[int]) -> int:
        ans, size = 0, len(A)
        min_right, max_left = [None] * size, [None] * size
        for i in range(size-1, -1, -1):
            if i == size-1:
                min_right[i] = A[i]
            else:
                min_right[i] = min(min_right[i+1],A[i])

        for i in range( size):
            if i == 0:
                max_left[i] = A[i]
            else:
                max_left[i] = max(max_left[i-1], A[i])
            if max_left[i] < min_right[i+1]:
                ans = i+1
                break
        return ans

    '''
    1363. 形成三的最大倍数
    数论
    '''
    def largestMultipleOfThree(self, digits: List[int]) -> str:
        sub_num = [[] for _ in range(2)]
        sum = 0
        for num in digits:
            sum += num
            if num % 3 == 1:
                sub_num[0].append(num)
            elif num % 3 == 2:
                sub_num[1].append(num)
        sub_num[0].sort()
        sub_num[1].sort()
        if sum % 3 == 1:
            if len(sub_num[0]) != 0:
                digits.remove(sub_num[0][0])
            elif len(sub_num[1]) >= 2:
                digits.remove(sub_num[1][0])
                digits.remove(sub_num[1][1])
            else:
                return ""
        elif sum % 3 == 2:
            if len(sub_num[1]) != 0:
                digits.remove(sub_num[1][0])
            elif len(sub_num[0]) >= 2:
                digits.remove(sub_num[0][0])
                digits.remove(sub_num[0][1])
            else:
                return ""
        ans = "".join(map(lambda x: str(x), sorted(digits,reverse=True)))
        if ans != "" and not int(ans):
            ans = '0'
        return ans

    '''
    841. 钥匙和房间
    dfs
    '''
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        size = len(rooms)
        ans,visited = 1, [False for _ in range(size)]
        visited[0] = True
        def dfs(x):
            nonlocal ans
            if ans == size:
                return
            for i in rooms[x]:
                if not visited[i]:
                    visited[i] = True
                    ans += 1
                    dfs(i)
        dfs(0)
        return True if ans == size else False

    '''
    486. 预测赢家
    动态规划
    '''
    def PredictTheWinner(self, nums: List[int]) -> bool:
        size = len(nums)
        pre_nums = [0]
        for i in range(size):
            pre_nums.append(pre_nums[-1]+nums[i])
        dp, tmp = nums[:], []
        for l in range(2, size+1):
            tmp.clear()
            for i in range(size - l + 1):
                tmp.append(pre_nums[i+l] - pre_nums[i] - min(dp[i], dp[i+1]))
            dp = tmp[:]
        return dp[0] >= pre_nums[-1]/2

    '''
    39. 组合总和
    动态规划
    '''
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = [[] for _ in range(501)]
        size = len(candidates)
        candidates = sorted(candidates)

        for num in range(candidates[0], target+1):
            if num in candidates:
                res[num].append([num])
            for i in range(size):
                if candidates[i] > num /2:
                    break
                else:
                    sub =  num - candidates[i]
                    if len(res[sub]) == 0:
                        continue
                    else:
                        for x in res[sub]:
                            t = x[:]
                            t.extend([candidates[i]])
                            # t = sorted(t)
                            # if t not in res[num]:
                            #     res[num].append(t)
                            res[num].append(t)
        return res[target]

    '''
    40. 组合总和 II
    '''
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates = sorted(candidates)
        res = {}

        def dfs(candidates, target):
            nonlocal res
            if target in res:
                return
            for num in candidates:
                if num > target:
                    break
                elif num == target:
                    if target in res:
                        if [num] not in res[target]:
                            res[target].append([num])
                    else:
                        res[target] = [[num]]
                elif num < target:
                    tmp = candidates[:]
                    tmp.remove(num)
                    dfs(tmp,target-num)
                    if target-num in res:
                        for nums in res[target-num]:
                            tmp = nums[:]
                            tmp.append(num)
                            tmp = sorted(tmp)
                            if target not in res:
                                res[target] = [tmp]
                            elif tmp not in res[target]:
                                res[target].append(tmp)

        dfs(candidates, target)
        print(res)
        return res[target]
    '''
    37. 解数独
    '''

    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        #--------回溯---------
        # def dfs(board: List[List[str]], x, y):
        #     if x >= len(board):
        #         return True
        #     x_n, y_n = x, y + 1
        #     if y_n >= len(board):
        #         x_n += 1
        #         y_n = 0
        #     if board[x][y] != '.':
        #         if dfs(board, x_n, y_n):
        #             return True
        #         else:
        #             return False
        #     # 验证
        #     for i in range(1, 10):
        #         flag = False
        #         # 行
        #         for col in range(9):
        #             if board[x][col] == str(i):
        #                 flag = True
        #                 break
        #         if flag: continue
        #         # 列
        #         for row in range(9):
        #             if board[row][y] == str(i):
        #                 flag = True
        #                 break
        #         if flag: continue
        #         # 九宫格
        #         xx, yy = x // 3 * 3, y // 3 * 3
        #         for row in range(xx, xx + 3):
        #             for col in range(yy, yy + 3):
        #                 if board[row][col] == str(i):
        #                     flag = True
        #                     break
        #             if flag: break
        #         if flag: continue
        #         board[x][y] = str(i)
        #         if dfs(board, x_n, y_n):
        #             return True
        #         else:
        #             board[x][y] = '.'
        #
        # dfs(board, 0, 0)
        # return board

        #-----------剪枝优化----------
        nums = [str(x) for x in range(1, 10)]
        column = [nums[:] for i in range(10)]
        row = [nums[:] for i in range(10)]
        block = [nums[:] for i in range(10)]
        empty = []
        for r in range(9):
            for c in range(9):
                num = board[r][c]
                if num == '.':
                    empty.append((r,c))
                else:
                    column[c].remove(num)
                    row[r].remove(num)
                    block[r//3*3 + c//3].remove(num)
        e_size = len(empty)

        def dfs(d):
            if d == e_size:
                return True
            x,y = empty[d]
            flag =False
            for num in range(1,10):
                num = str(num)
                if num in row[x] and num in column[y] and num in block[x//3*3 + y//3]:
                    board[x][y] = num
                    column[y].remove(num)
                    row[x].remove(num)
                    block[x // 3 * 3 + y // 3].remove(num)
                    if dfs(d+1):
                        return True
                    column[y].append(num)
                    row[x].append(num)
                    block[x // 3 * 3 + y // 3].append(num)
        dfs(0)
        return board

    '''
    685. 冗余连接 II
    有向图 判环
    '''
    def findRedundantDirectedConnection(self, edges: List[List[int]]) -> List[int]:
        # 数据预处理
        res,size = None, len(edges)
        points,visited = [[[],[]] for _ in range(size+1)],[False for _ in range(size+1)]
        for edge in edges:
            x,y = edge
            points[x][0].append(y)
            points[y][1].append(x)
            # 当一个节点有两个父节点，则做标记，肯定在这拆解
            if len(points[y][1]) == 2:
                res = y
        def find_loop(node):
            pre_note = points[node][1][0]
            while len(points[pre_note][1]):
                tmp_note = points[pre_note][1][0]
                if tmp_note == node:
                    return points[node][1][0]
                pre_note = tmp_note
            return points[node][1][1]
        def find_endnote(first_note):
            que = deque()
            que.append(first_note)
            while len(que):
                point = que.popleft()
                visited[point] = True
                nexts = points[point][0]
                for n in nexts:
                    if visited[n] == True:
                        return point
                    que.append(n)
            for i in range(1,size+1):
                if not visited[i]:
                    return find_endnote(i)
            return None

        if res is not None:
            return [find_loop(res), res]

        end_note = find_endnote(edges[0][0])
        pre_note = points[end_note][1][0]
        print(end_note)

        ans = [points[end_note][1][0], end_note]
        idx = edges.index(ans)

        while pre_note != end_note:
            tmp_note = points[pre_note][1][0]
            tmp_ans = [tmp_note,pre_note]
            tmp_idx = edges.index(tmp_ans)
            if tmp_idx > idx:
                idx, ans = tmp_idx, tmp_ans
            pre_note = tmp_note
        return ans

    '''
    416. 分割等和子集
    '''
    def canPartition(self, nums: List[int]) -> bool:
        v = sum(nums)
        if v % 2: return False
        v,size = v//2, len(nums)
        #--------------------------
        #       01 背包 一维
        dp = [ 0 for _ in range(v+1)]
        for i in range(size):
            for j in range(v, nums[i]-1, -1):
                if j >= nums[i]:
                    dp[j] = max(dp[j],dp[j-nums[i]] + nums[i])

        #--------------------------
        #       01 背包 二维
        # dp = [ [0] * (v+1) for _ in range(size)]
        # for i in range(size):
        #     for j in range(nums[i], v+1):
        #             dp[i][j] = max(dp[i-1][j],dp[i-1][j-nums[i]] + nums[i])
        # if dp[size-1][v] == v:
        #     return True
        # return False

        print(dp)
        if dp[v] == v:
            return True
        return False

    '''
    399. 除法求值
    '''

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        uf = UnionFind()
        for [a,b],c in zip(equations, values):
            uf.add(a)
            uf.add(b)
            uf.merge(a,b,c)

        ans = [-1.0] * len(queries)
        for i, [x, y] in enumerate(queries):
            if x in uf.values and y in uf.values and uf.is_connected(x, y):
                ans[i] = uf.father[x][1] / uf.father[y][1]

        return ans

class UnionFind:
    def __init__(self):
        self.father = {}
        self.values = set()

    def find(self, x):
        if self.father[x][0] == x: return self.father[x]
        resPre = self.find(self.father[x][0])
        print(x)
        print(self.father[x][0], self.father[x][1])
        print(resPre[0], resPre[1])
        self.father[x] = [resPre[0], self.father[x][1] * resPre[1]]
        return self.father[x]

    def merge(self, x, y, val):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.father[root_x[0]] = [root_y[0], val *(1/ root_x[1]) * root_y[1]]

    def is_connected(self, x, y):
        return self.find(x)[0] == self.find(y)[0]

    def add(self, x):
        if x not in self.father:
            self.father[x] = [x, 1]
            self.values.add(x)

if __name__ == "__main__":
    sol = Solution()
    # 67
    # print(sol.addBinary("11","1"))

    #268
    # print(sol.missingNumber(nums = [3,0,1]))

    #16
    # nums = [-1,2,1,4]
    # print(sol.threeSumClosest(nums, 1))

    #139
    # s = "applepenapple"
    # wordDict = ["apple", "pen"]
    # print(sol.wordBreak(s,wordDict))

    #5
    # print(sol.longestPalindrome("a"))

    #209
    # print(sol.minSubArrayLen(7,[2,3,1,2,4,3]))

    #718
    # print(sol.findLength([1,2,3,2,1],[3,2,1,4,7]))

    #63
    # print(sol.uniquePathsWithObstacles([[0,0,0],[0,1,0],[0,0,0]]))

    #108
    # def printTree(tree: TreeNode):
    #     if tree:
    #         print(tree.val, end=" ")
    #     else:
    #         print("null",end=" ")
    #         return
    #     printTree(tree.left)
    #     printTree(tree.right)
    # tree = sol.sortedArrayToBST([-10,-3,0,5,9])
    # printTree(tree)

    #112
    # tree = TreeNode(5)
    # print(sol.hasPathSum(tree))

    #面试题 17.13. 恢复空格
    # print(sol.respace(["looked","just","like","her","brother"], "jesslookedjustliketimherbrother"))

    #11
    # print(sol.maxArea([1,8,6,2,5,4,8,3,7]))

    #6
    # print(sol.convert("LEETCODEISHIRING",4))

    #315
    # print(sol.countSmaller([5,2,6,1]))

    #174
    # print(sol.calculateMinimumHP([[0,-9]]))

    #96
    # print(sol.numTrees(5))

    #785
    # print(sol.isBipartite( [[1,2,3], [0,2], [0,1,3], [0,2]]))

    #35
    # print(sol.searchInsert([1,3,5,6], 5))

    #312
    # print(sol.maxCoins([3,1,5,8]))

    #95
    # def printTree(tree: TreeNode):
    #     if tree:
    #         print(tree.val, end=" ")
    #     else:
    #         print("null",end=" ")
    #         return
    #     printTree(tree.left)
    #     printTree(tree.right)
    # trees = sol.generateTrees(3)
    # for tree in trees:
    #     printTree(tree)
    #     print()

    #42
    # print(sol.trap([4,2,3]))

    #592
    # print(sol.fractionAddition("-1/2+1/2"))

    #1025
    # print(sol.divisorGame(1000))

    #410
    # print(sol.splitArray([7,2,5,10,8],5))

    #343
    # print((sol.integerBreak(10)))

    #336
    # print(sol.palindromePairs(["bb","bababab","baab","abaabaa","aaba","","bbaa","cabac","baa","b"]))

    #207
    # print(sol.canFinish(2, [[1,0]]))

    #529
    # print(sol.updateBoard(
    #     [["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"],
    #      ["E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E",
    #       "E", "E", "E", "E", "E", "E"]],
    #     [29, 2]
    #
    # ))

    #647
    # print(sol.countSubstrings("abaaba"))

    #679
    # print(sol.judgePoint24([3,3,7,7]))

    #201
    # print(sol.rangeBitwiseAnd(5,9))

    #491
    # print(sol.findSubsequences([4,6,5,7]))

    #17
    # print(sol.letterCombinations("23"))

    #915
    # print(sol.partitionDisjoint([1,1,1,0,6,12]))

    #1363
    # print(sol.largestMultipleOfThree([0,0,0,0]))

    #841
    # print(sol.canVisitAllRooms([[0]]))

    #486
    # print(sol.PredictTheWinner([1,5,2,7,4]))

    #39
    # print(sol.combinationSum([2,3,4,5,6],8))

    #40
    # print(sol.combinationSum2([4,4,2,1,4,2,2,1,3],6))

    #37
    # print(sol.solveSudoku([["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]))

    #685
    # print(sol.findRedundantDirectedConnection([[2,1],[4,2],[1,4],[3,1]]))

    #416
    # print(sol.canPartition([1, 2, 5]))

    #399

    print(sol.calcEquation(

[["x1","x2"],["x2","x3"],["x1","x4"],["x2","x5"]],
[3.0,0.5,3.4,5.6],
[["x2","x4"],["x1","x5"],["x1","x3"],["x5","x5"],["x5","x1"],["x3","x4"],["x4","x3"],["x6","x6"],["x0","x0"]]

))