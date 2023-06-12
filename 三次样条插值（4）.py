
x = []
y = []
'''
for i in range(21):
    g = i/20
    x.append(g)
    y.append(1/(1+25*g*g))
'''
for i in range(21):
    g = (i*i)/(20*20)
    x.append(g)
    y.append(1 / (1 + 25 * g * g))   # 生成测试数据

'''
x = eval(input())
y = eval(input())
'''
h = []
for i in range(len(x)-1):
    h.append(x[i+1]-x[i])
a = [[0]*len(x) for i in range(len(x))]
a[0][0] = 1
a[0][1] = 1
a[len(x)-1][len(x)-2] = 1
a[len(x)-1][len(x)-1] = 1
for i in range(1, len(h)):
    a[i][i-1] = h[i]/(h[i-1]+h[i])
    a[i][i] = 2
    a[i][i+1] = h[i-1] / (h[i - 1] + h[i])
b = [0]*len(x)
b[0] = 2*(y[1]-y[0])/h[0]
b[len(x)-1] = 2*(y[len(y)-1]-y[len(y)-2])/h[len(x)-2]
for i in range(1, len(x)-1):
    b[i] = 3*(h[i]*(y[i]-y[i-1])/(h[i-1]*(h[i]+h[i-1]))+h[i-1]*(y[i+1]-y[i])/(h[i]*(h[i]+h[i-1])))
m = []


def det(matric):
    if len(matric) <= 0:
        return None
    elif len(matric) == 1:
        return matric[0][0]
    else:
        s = 0
        for o in range(len(matric)):
            if matric[0][o] == 0 :
                continue
            n = [[row[p] for p in range(len(matric)) if p != o] for row in matric[1:]]  # 这里生成余子式
            s += matric[0][o] * det(n) * (-1) ** (o % 2)
        return s


import copy
for i in range(len(x)):
    a_ = copy.deepcopy(a)
    for j in range(len(x)):
        a_[j][i] = b[j]
    m.append(det(a_)/det(a))
l = [[0]*4 for g in range(len(x)-1)]
for i in range(len(l)):
    l[i][0] = ((2*(y[i]-y[i+1])/h[i])+m[i]+m[i+1])/(h[i]*h[i])
    l[i][1] = ((y[i]*(h[i]-2*x[i]-4*x[i+1])+y[i+1]*(h[i]+2*x[i+1]+4*x[i]))/h[i]-m[i]*(x[i]+2*x[i+1])-m[i+1]*(x[i+1]+2*x[i]))/(h[i]*h[i])
    l[i][2] = ((y[i]*(2*x[i+1]*x[i+1]-2*x[i+1]*(h[i]-2*x[i]))+y[i+1]*(-2*x[i]*x[i]-2*x[i]*(h[i]+2*x[i+1])))/h[i]+m[i]*(2*x[i]*x[i+1]+x[i+1]*x[i+1])+m[i+1]*(2*x[i]*x[i+1]+x[i]*x[i]))/(h[i]*h[i])
    l[i][3] = ((y[i]*x[i+1]*x[i+1]*(h[i]-2*x[i])+y[i+1]*x[i]*x[i]*(h[i]+2*x[i+1]))/h[i]-x[i]*x[i+1]*(m[i]*x[i+1]+m[i+1]*x[i]))/(h[i]*h[i])

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,1,100)
y0 = [1]
y1 = []
'''
for i in t:
    for j in range(20):
        if (i>(j/20)) and (i <= ((j + 1) / 20)):
            y0_ = l[j][0]*i**3 + l[j][1]*i**2 + l[j][2]*i + l[j][3]
            y0.append(y0_)
'''
for i in t:
    for j in range(20):
        if (i>((j*j)/(20*20))) and (i <= (((j + 1)*(j + 1))/(20*20))):
            y0_ = l[j][0]*i**3 + l[j][1]*i**2 + l[j][2]*i + l[j][3]
            y0.append(y0_)

for i in t:
    y1_ = 1/(1+25*i**2)
    y1.append(y1_)
y_ = [y0[i]-y1[i] for i in range(len(y0))]
plt.plot(t,y_)
plt.xlabel('x')
plt.ylabel('y*-y')
plt.show()
for j in range(20):
    if (0.03 > (j / 20)) and (0.03 <= ((j + 1) / 20)):
        print('y*-y|0.03 = ',l[j][0]*0.03**3 + l[j][1]*0.03**2 + l[j][2]*0.03 + l[j][3]-1/(1+25*0.03**2))
    elif (0.97 > (j / 20)) and (0.97 <= ((j + 1) / 20)):
        print('y*-y|0.97 = ',l[j][0]*0.97**3 + l[j][1]*0.97**2 + l[j][2]*0.97 + l[j][3]-1/(1+25*0.97**2))
