try:
    import matplotlib.pyplot as plt     #提供和matlab类似的绘图API
    # import matplotlib.pyplot as pltp
    import numpy as np
except ImportError:
    pass

# #1 基础绘图
# #第1步：定义x和y坐标轴上的点   x坐标轴上点的数值
# x=[1, 2, 3, 4]
# #y坐标轴上点的数值
# y=[1, 4, 9, 16]
# #第2步：使用plot绘制线条第1个参数是x的坐标值，第2个参数是y的坐标值
# plt.plot(x,y)
# #第3步：显示图形
# # plt.show()
#
# #添加文本 #x轴文本
# plt.xlabel('xlabel')
# #y轴文本
# plt.ylabel('ylabel')
# #标题
# plt.title('title')
# #添加注释 参数名xy：箭头注释中箭头所在位置，参数名xytext：注释文本所在位置，
# #arrowprops在xy和xytext之间绘制箭头, shrink表示注释点与注释文本之间的图标距离
#
# plt.annotate('i am zs', xy=(2,5), xytext=(2, 10),
#             arrowprops=dict(facecolor='black', shrink=0.01),
#             )
#
# #2 定义绘图属性
# '''
# color：线条颜色，值r表示红色（red）
# marker：点的形状，值o表示点为圆圈标记（circle marker）
# linestyle：线条的形状，值dashed表示用虚线连接各点
# '''
# plt.plot(x, y, color='r',marker='o',linestyle='dashed')
# #plt.plot(x, y, 'ro')
# '''
# axis：坐标轴范围
# 语法为axis[xmin, xmax, ymin, ymax]，
# 也就是axis[x轴最小值, x轴最大值, y轴最小值, y轴最大值]
# '''
# plt.axis([0, 6, 0, 20])
# # plt.show()
#
# x1 = [1, 2, 3, 4]
# y1 = [1, 4, 9, 16]
# x2 = [1, 2, 3, 4]
# y2 = [2, 4, 6, 8]
#
# plt.plot(x1, y1, label='Line 1')
# plt.plot(x2, y2, label='Line 2')
#
# # plt.legend()
# plt.legend(handlelength=5, handletextpad=0.9)
# plt.show()
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np
# x = np.arange(10)
x = [0.140, 0.150, 0.160, 0.170, 0.180]
#mwpm 码距9
y = [0.32, 0.385, 0.46, 0.54, 0.62]
plt.scatter(x, y, marker='o', color='red', s=100)
plt.plot(x, y, color='red', label='L = 9')

y = [0.323, 0.383, 0.452, 0.521, 0.575]
plt.scatter(x, y, marker='o', color='green', s=100)
plt.plot(x, y, color='green', label='L = 7')

y = [0.335, 0.38, 0.447, 0.51, 0.55]
plt.scatter(x, y, marker='o', color='blue', s=100)
plt.plot(x, y, color='blue', label='L = 5')

#nn
y = [0.26, 0.307, 0.375, 0.44, 0.503]
plt.scatter(x, y, marker='*', color='green', s=100)
plt.plot(x, y, color='green')

y = [0.285, 0.326, 0.38, 0.436, 0.485]
plt.scatter(x, y, marker='*', color='blue', s=100)
plt.plot(x, y, color='blue')

plt.scatter(x, y, marker='o', label='mwpm', color='black')
plt.scatter(x, y, marker='*', label='nn', color='black')
plt.legend()
# y = [0.22, 0.29, 0.33]
# plt.plot(x, y)
# plt.plot(x, x)
# plt.plot(x, 2 * x)
# plt.plot(x, 3 * x)
# plt.plot(x, 4 * x)


plt.xlabel("Depolarization Error Rate")
plt.ylabel("Logical Error Rate")
plt.show()