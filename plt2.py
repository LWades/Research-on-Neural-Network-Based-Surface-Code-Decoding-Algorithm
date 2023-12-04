# try:
#     import matplotlib.pyplot as plt     #提供和matlab类似的绘图API
#     # import matplotlib.pyplot as pltp
#     import numpy as np
# except ImportError:
#     pass

from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np
# x = np.arange(10)
x = [0.140, 0.150, 0.160, 0.170, 0.180]

#nn
y = [0.26, 0.307, 0.375, 0.44, 0.503]
plt.scatter(x, y, marker='*', color='green', s=100)
plt.plot(x, y, color='green', label='L = 7')

y = [0.285, 0.326, 0.38, 0.436, 0.485]
plt.scatter(x, y, marker='*', color='blue', s=100)
plt.plot(x, y, color='blue', label='L = 5')

#平移对称性
y = [0.20, 0.247, 0.315, 0.38, 0.443]
plt.scatter(x, y, marker='^', color='green', s=100)
plt.plot(x, y, color='green')

y = [0.215, 0.256, 0.327, 0.375, 0.425]
plt.scatter(x, y, marker='^', color='blue', s=100)
plt.plot(x, y, color='blue')

#轴对称性
y = [0.24, 0.277, 0.365, 0.41, 0.473]
plt.scatter(x, y, marker='p', color='green', s=100)
plt.plot(x, y, color='green')

y = [0.265, 0.296, 0.370, 0.407, 0.475]
plt.scatter(x, y, marker='p', color='blue', s=100)
plt.plot(x, y, color='blue')

plt.scatter(x, y, marker='*', label='nn without symmetry', color='black')
plt.scatter(x, y, marker='^', label='translation symmetry', color='black')
plt.scatter(x, y, marker='p', label='axial symmetry', color='black')
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