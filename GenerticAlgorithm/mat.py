from math import inf
from random import *

vol = [[174, 538], [1287, 1345], [229, 129], [177, 159], [171, 293], [1245, 1300], [206, 381], [192, 182], [301, 403],
       [327, 463], [164, 303], [408, 359]]
juli = [0.0, 1.256, 0.568, 1.15, 0.895, 1.039, 4.527, 4.281, 5.478, 5.894, 3.048, 8.743, 2.307, 1.982, 1.256, 0.0,
        1.749, 2.458, 1.35, 2.35, 4.964, 4.384, 5.015, 5.667, 4.207, 6.432, 2.377, 3.014, 0.568, 1.749, 0.0, 0.67,
        1.268, 0.647, 3.725, 3.021, 4.319, 4.012, 2.819, 5.284, 1.805, 1.091, 1.15, 2.458, 0.67, 0.0, 1.992, 0.873,
        3.118, 3.268, 3.206, 3.89, 2.202, 4.469, 1.315, 1.277, 0.895, 1.35, 1.268, 1.992, 0.0, 1.796, 3.946, 4.237,
        4.263, 4.954, 3.264, 5.526, 1.591, 2.444, 1.039, 2.35, 0.647, 0.873, 1.796, 0.0, 6.177, 3.076, 3.961, 4.522,
        2.978, 5.301, 2.002, 1.286, 4.527, 4.964, 3.725, 3.118, 3.946, 6.177, 0.0, 1.869, 0.678, 1.444, 0.524, 2.061,
        2.018, 3.427, 4.281, 4.384, 3.021, 3.268, 4.237, 3.076, 1.869, 0.0, 1.256, 0.701, 1.092, 1.592, 1.756, 1.0,
        0.429, 5.478, 5.015, 4.319, 3.206, 4.263, 3.961, 0.678, 1.256, 0.0, 0.585, 0.96, 1.375, 2.546, 3.716, 3.048,
        5.667, 4.012, 3.89, 4.954, 4.522, 1.444, 0.701, 0.585, 0.0, 1.59, 0.954, 3.061, 4.284, 8.743, 4.207, 2.819,
        2.202, 3.264, 2.978, 0.524, 1.092, 0.96, 1.59, 0.0, 2.482, 1.463, 2.588, 2.307, 6.432, 5.284, 4.469, 5.526,
        5.301, 2.061, 1.592, 1.375, 0.954, 2.482, 0.0, 2.588, 3.321, 1.982, 2.377, 1.805, 1.315, 1.0, 0.591, 2.002,
        2.018, 1.756, 2.546, 3.061, 1.463, 2.588, 0.0, 1.605, 3.014, 3.014, 1.091, 1.277, 2.0, 444.0, 1.286, 3.427,
        1.429, 3.716, 4.284, 2.588, 3.321, 1.605, 0.0]
D_max = 10
Pc = 0.6    # 交叉概率
Pm = 0.04   # 变异概率

x = []
for i in range(14):
    x.append(juli[i*14: (i+1)*14])
juli = x


def ga_generation():
    global vol, juli, D_max
    M = [0,1,2,3,4,5,6,7,8,9,10,11]
    N = [12,13]
    x = []
    x1 = []

    while M:
        route_length = 0
        route_vol = 0
        e1 = randint(0,1)
        z1 = N[e1]
        z2 = [z1]
        a = 0
        e2 = 0

        while route_length <= D_max:
            print(x1)
            if route_vol <= 1400:
                z2 = [a, *z2]
                if e2 != 0:
                    M[e2:e2+1] = []
                if M:
                    e2 = randint(0, len(M)-1)
                    k = 0
                    h = 0
                    for j in range(len(z2)-1):
                        if len(z2) == 1:
                            k = 0
                            h = 0
                        else:
                            print(z2, j)
                            k1 = juli[z2[j]][z2[j+1]]
                            k = k + k1
                            h1 = vol[M[e2]][e1]
                            h = h + h1
                    a = M[e2]
                    route_length = k
                    route_vol = h
                else:
                    break
            x1.append(z2)

        x2 = x1
        g = 22 - len(x1)
        for i in range(g):
            e3 = randint(2, len(x1)-1)
            a1 = x1[1:e3]
            a2 = x1[e3+1:]
            x1 = [a1, 0, a2]

    return x1



def fitness(b):
    global vol, juli
    Vr = 40
    Vb = 18
    FR = 12
    a = 0
    k = 0

    Fk = 0

    for i in range(22):
        if b[i] == 13:
            c = b[k+1][i]
            k = i
            sum1 = 0
            sum2 = 0
            for j in range(len(c)-1):
                if c[j] != 0:
                    sum1 = sum1 + vol[c[j]][0]
                    sum2 = sum2 + juli[c[j]][13]

                    Fk = sum1 / 40
            a = a+(1/2*Fk)*sum1 + (1/2*FR)*sum1 + sum1*sum2/Vb + 1.605*sum1/Vr + 5.2*Fk*sum2
        elif b[i] == 14:
            c = b[k+1][i]
            k = i
            sum1 = 0
            sum2 = 0
            for j in range(len(c)-1):
                if c[j] != 0:
                    sum1 = sum1 + vol[c[j]][2]
                    sum2 = sum2 + juli[c[j]][14]
                    Fk = sum1 / 40
            a = a+(1/2*Fk)*sum1 + (1/2*FR)*sum1 + sum1*sum2/Vb + 1.605*sum1/Vr + 5.2*Fk*sum2
    a = -a
    return a

def cross(A, B):
    global D_max, juli, vol
    A[A==0] = []
    B[B==0] = []
    A_chose = A[1]
    k = 1
    D = []
    M1 = A
    M2 = B
    F = []
    DD = []

    while len(DD) < 12:
        route_length = 0
        route_vol = 0
        C = []

        while route_length <= D_max:
            if route_vol <= 1400:
                C = [C, A_chose]
                if len(C) == 1:
                    for i in range(k, len(A)+1):
                        if A[i] == 13:
                            k2 = 1
                            b = A[i]
                            break
                        elif A[i] == 14:
                            k2 = 2
                            b = A[i]
                            break
            if k == 1:
                if M1[k+1] == 13 or M1[k+1] == 14:
                    S4 = []
                else:
                    S4 = M1[k+1]
                S1 = S4
            elif k == len(A):
                if M1[k-1] == 13 or M1[k-1] == 14:
                    S3 = []
                else:
                    S3 = M1[k-1]
                S1 = S3
            else:
                S3 = M1[k-1]
                S4 = M1[k+1]
                if M1[k + 1] == 13 or M1[k + 1] == 14:
                    S4 = []
                if M1[k - 1] == 13 or M1[k - 1] == 14:
                    S3 = []
                S1 = [S3, S4]
            k1 = M2.index(A_chose)
            if k1 == 1:
                if M2[k1 + 1] == 13 or M2[k1 + 1] == 14:
                    S6 = []
                else:
                    S6 = M2[k1 + 1]
                S2 = S6
            elif k1 == len(M2):
                if M2[k1 - 1] == 13 or M2[k1 - 1] == 14:
                    S5 = []
                else:
                    S5 = M2[k1 - 1]
                S2 = S5
            else:
                S5 = M1[k1 - 1]
                S6 = M1[k1 + 1]
                if M2[k1 + 1] == 13 or M2[k1 + 1] == 14:
                    S6 = []
                if M1[k1 - 1] == 13 or M1[k1 - 1] == 14:
                    S5 = []
                S2 = [S5, S6]
            S = [S1, S2]
            h1 = []
            D = []
            D = [DD, C]
            if D:
                for i in range(S):
                    for j in range(D):
                        if S[i] == D[j]:
                            h1 = [h1, i]
            S[h1] = []
            if S:
                S_juli = []
                for i in range(S):
                    S_juli[i] = juli[A_chose][S[i]]
            S_min = m = min(S_juli)
            A_chose = S[m]
            C1 = [C,A_chose,b]
            route_length = 0
            route_vol = 0
            for i in range(len(C1)-1):
                route_length = route_length + juli(C1[i], C1[i+1])
                route_vol = route_vol + vol[C1[i]][k2]
        else:
            for i in range(A):
                r = []
                if A[i]!=13 and A[i]!=14:
                    for j in range(D):
                        # r.append()
    #                      r = [r find(A[i] == D[j])
                        pass
                    if r == 0:
                        A_chose = A[i]
                        break
        k = M1.index(A_chose)
        if len(D) == 12:
            break
        else:
            break
    DD.append(C)
    F.extend([C, b])
    E = A
    for i in range(E):
        r = []
        if E[i]!=13 and E[i]!=14:
            for j in range(len(DD)):
                # r.append()
                # r = [r find(A[i] == D[j])
                pass
            if r:
                A_chose = A[i]
                break
    g = 22 - len(F)
    for i in range(g):
        e3 = randint(2, len(F)-1)
        a1 = F[1:e3]
        a2 = F[e3+1:]
        F = [a1, 0, a2]
    return F

def myGA():
    global vol, juli, NP, D_max, NG, Pc, Pm
    ksi0 = 2
    c = 0.9
    x = [0] * NP
    fx = [0] * NP
    SelFather = 0
    nx = [0]
    xv = 0
    fv = 0

    # 生成结果 并 求得适应度
    for i in range(NP):
        x[i] = ga_generation()
        fx[i] = fitness(x[i])

    ksi = ksi0
    for k in range(NG):
        # 求选择概率
        fmin = min(fx)
        Normfx = [ a - 1 + ksi  for a in fx]
        sumfx = sum(Normfx)
        Px = [a / sumfx  for a in Normfx]

        PPx = [0] * NP
        PPx[0] = Px[0]
        for i in range(1, NP):
            PPx[i] = PPx[i-1] + Px[i]

        # 交叉
        for i in range(NP):
            sita = random()
            for n in range(NP):
                if sita <= PPx[n]:
                    SelFather = n
                    break
            SelMother = randint(0, NP-1)
            r1 = random()
            if r1 <= Pc:
                nx[i] = cross(x[SelFather], x[SelMother])
                r2 = random()
                if r2 <= Pm:
                    m1 = [randint(2, 21),randint(2, 21)]
                    nx[i][m1[0]], nx[i][m1[1]] = nx[i][m1[1]], nx[i][m1[0]]
            else:
                nx[i] = x[SelFather]
        x = nx

        for i in range(NP):
            fx[i] = fitness(x[i])
            ksi = ksi * c

    fx = -inf
    for i in range(NP):
        fitx = fitness(x[i])
        if fitx > fx:
            fv = fitx
            xv = x[i]

    fv = -fv
    return [xv, fv]



if __name__ == "__main__":
    print(ga_generation())