import math
from random import *
import numpy as np

vol = [[174,538],[1287,1345],[229,129],[177,159],[171,293],[1245,1300],[206,381],[192,182],[301,403],[327,463],[164,303],[408,359]]
juli = [0.0, 1.256, 0.568, 1.15, 0.895, 1.039, 4.527, 4.281, 5.478, 5.894, 3.048, 8.743, 2.307, 1.982, 1.256, 0.0, 1.749, 2.458, 1.35, 2.35, 4.964, 4.384, 5.015, 5.667, 4.207, 6.432, 2.377, 3.014, 0.568, 1.749, 0.0, 0.67, 1.268, 0.647, 3.725, 3.021, 4.319, 4.012, 2.819, 5.284, 1.805, 1.091, 1.15, 2.458, 0.67, 0.0, 1.992, 0.873, 3.118, 3.268, 3.206, 3.89, 2.202, 4.469, 1.315, 1.277, 0.895, 1.35, 1.268, 1.992, 0.0, 1.796, 3.946, 4.237, 4.263, 4.954, 3.264, 5.526, 1.591, 2.444, 1.039, 2.35, 0.647, 0.873, 1.796, 0.0, 6.177, 3.076, 3.961, 4.522, 2.978, 5.301, 2.002, 1.286, 4.527, 4.964, 3.725, 3.118, 3.946, 6.177, 0.0, 1.869, 0.678, 1.444, 0.524, 2.061, 2.018, 3.427, 4.281, 4.384, 3.021, 3.268, 4.237, 3.076, 1.869, 0.0, 1.256, 0.701, 1.092, 1.592, 1.756, 1.0, 0.429, 5.478, 5.015, 4.319, 3.206, 4.263, 3.961, 0.678, 1.256, 0.0, 0.585, 0.96, 1.375, 2.546, 3.716, 3.048, 5.667, 4.012, 3.89, 4.954, 4.522, 1.444, 0.701, 0.585, 0.0, 1.59, 0.954, 3.061, 4.284, 8.743, 4.207, 2.819, 2.202, 3.264, 2.978, 0.524, 1.092, 0.96, 1.59, 0.0, 2.482, 1.463, 2.588, 2.307, 6.432, 5.284, 4.469, 5.526, 5.301, 2.061, 1.592, 1.375, 0.954, 2.482, 0.0, 2.588, 3.321, 1.982, 2.377, 1.805, 1.315, 1.0, 0.591, 2.002, 2.018, 1.756, 2.546, 3.061, 1.463, 2.588, 0.0, 1.605, 3.014, 3.014, 1.091, 1.277, 2.0, 444.0, 1.286, 3.427, 1.429, 3.716, 4.284, 2.588, 3.321, 1.605, 0.0]
D_max = 10


def fitness(b):
    global vol, juli
    Vr=40
    Vb=18
    FR=12
    a=0
    k=0
    Fk = 0
    for i in range(1, 23):
      if b[i]==13:
        c=b[k+1][i]
        k=i
        sum1=0
        sum2=0
        for j in range(1,len(c)):
          if c[j] != 0:
            sum1 = sum1 + vol[c[j]][1]
            sum2 = sum2 + juli[c[j]][13]
            Fk = sum1/40
        a = a + (1/2 * Fk) * sum1 +(1/2*FR)*sum1 + sum1 *sum2/Vb+ 1.605*sum1/Vr+5.2 *Fk*sum2
      elif b[i] == 14:
        c = b[k+1][i]
        k=i
        sum1=0
        sum2=0
        for j in range(1, len(c)):
          if c[j]!=0:
            sum1 = sum1 + vol[c[j]][2]
            sum2 = sum2 + juli[c[j]][14]
            Fk= sum1 /40
        a=a+(1/2 *Fk)*sum1 +(1/2*FR)*sum1 +sum1 *sum2/Vb+1.605*sum1/Vr+5.2*Fk*sum2
    a=-a
    return a

def ga_generation():
    global val, juli, D_max
    M = [1,2,3,4,5,6,7,8,9,10,11,12]
    N = [13,14]
    x = []
    x1 = []
    while M:
        route_length = 0
        route_vol = 0
        e1 = round(random())
        z1 = N[e1]
        z2 = z1
        a = []
        e2 = 0
        while route_length <= D_max:
            if route_vol <= 1400:
                z2=[a,z2]
            if e2!=0:
                M[e2]=[]
            if M:
                e2=math.floor(random() * len(M) + 1)
                k=0
                h=0
                for j in range(1, len(z2)):
                    if len(z2) == 1:
                        k = 0
                        h = 0
                    else:
                        k1 = juli[z2[j]][z2[j + 1]]
                        k = k + k1
                        h1 = vol[M[e2]][e1]
                        h = h + h1
                a = M[e2]
                route_length = k
                route_vol = h
            else:
                break
        x1 = [x1, z2]
    x2 = x1
    g = 22 - len(x1)
    for i in range(1, g+1):
        e3 = randint(2, len(x1) - 1)
        a1 = x1[1:e3]
        a2 = x1[e3+1:]
        x1 = [a1, 0, a2]
    return x1

# def myga():
#     NP = 50
#     NG = 800
#     Pc = 0.6
#     Pm = 0.04
#     ksi0 = 2
#     c = 0.9
#
#     # _________
#     fx = []
#     x = []
#
#     for i in range(1,NP+1):
#         x[i] = ga_generation()
#         fx[i] = fitness(x[i])
#
#     ksi = ksi0
#     for k in range(1,NG+1):
#         fmin = min(fx)
#         Normfx = np.array(fx) - fmin*[[1]*len(fx) for _ in range(len(fx))] + ksi
#         sumfx = sum(Normfx)
#         Px = Normfx / sumfx
#         PPx = 0
#         # PPx(1) = Px(1)
#         for i in range(2, NP +1):
#             # PPx(i) = PPx(i-1) + Px(i)
#             pass
#         for i in range(1, NP+1):
#             sita = random()
#             for n in range(1, NP+1):
#                 if sita <= PPx[n]:
#                     SelFather = n;
#                     break
#             Selmother = math.floor(random() * (NP-1)) + 1
#             r1 = random()
#             if r1 <= Pc:
#                 nx = jiaocha(x(SelFather), )

if __name__ == "__main__":
    print(round(random()))





