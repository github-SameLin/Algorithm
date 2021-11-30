import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

def getAvgTime(TIME_PIECE, SWITCH_TIME, process_time):
    process_num = len(process_time)
    turnover_time = [0] * process_num
    cur_time = 0
    while True:
        solved_num = 0
        for i in range(process_num):
            if process_time[i]:
                t_time = TIME_PIECE
                # 剩余时间小于时间片， 计算周转时间
                if process_time[i] <= t_time:
                    t_time = process_time[i]
                    turnover_time[i] = cur_time + t_time
                # 结束时间 + 进程切换的时间
                cur_time += t_time + SWITCH_TIME
                process_time[i] -= t_time
            else:
                solved_num += 1
        if solved_num == process_num:
            break
    return sum(turnover_time) / process_num

x2=range(1,31)
y2=[getAvgTime(x, 0, [20,10,15,5]) for x in x2]

plt.plot(x2,y2)
plt.xlabel('时间片')
plt.ylabel('平均周转时间')
plt.title('0时间到达 / A:20ms -> B:10ms -> C:15ms -> D:5ms / 切换消耗：0ms')
plt.legend()
plt.show()