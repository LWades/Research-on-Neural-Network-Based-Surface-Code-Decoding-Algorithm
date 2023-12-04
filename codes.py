'''
Tested only in interactive use with the Jupyter notebook.
Some tools might fail without it due to the use of `tnrange`
and `tqdm_notebook` from `tqdm`. Similarly `matplotlib` is
used with its interactive interface, which might cause trouble
if `ioff` is not called.
整个code.py相当于一个工具包，其他的文件需要做什么从code中import函数来实现
'''


import itertools

import numpy as np                      #扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
import scipy.linalg                     #开源Python算法库和数学工具包scipy的线性代数包，包含numpy.linalg的所有函数
import scipy.stats as stats             #统计推断包
import scipy.optimize as optimize       #优化器，提供了常用的最优化算法函数实现，比如找函数的
# matplotlib.use('TKAgg')#new
try:
    import matplotlib.pyplot as plt     #提供和matlab类似的绘图API
    # import matplotlib.pyplot as pltp
except ImportError:
    pass

import networkx as nx                   #可以实现各种图算法
#NetworkX是一个Python软件包，用于创建、操作和研究复杂网络、图和相关结构的软件包。它包含了许多经典和现代的网络算法，可以用于网络分析、社交网络、生物网络、交通网络等方面的建模和分析。它支持有向图、无向图、加权图、多重图等多种类型的图形，还提供了丰富的图形可视化工具和数据结构的输入输出功能。

from tqdm import tqdm, trange           #进度条

try:
    from IPython import display         #可视化的
except ImportError:
    pass


class ToricCode:
    '''

    ::

        Lattice:
        X00--Q00--X01--Q01--X02...
         |         |         |
        Q10  Z00  Q11  Z01  Q12
         |         |         |
        X10--Q20--X11--Q21--X12...
         .         .         .
    ::

    举个例子
    L = 3的时候
        Lattice:
        X00--Q00--X01--Q01--X02--Q02--
         |    |    |    |    |    |
        Q10--Z00--Q11--Z01--Q12--Z02--
         |    |    |    |    |    |
        X10--Q20--X11--Q21--X12--Q22--
         |    |    |    |    |    |
        Q30--Z10--Q31--Z11--Q32--Z12--
         |    |    |    |    |    |
        X20--Q40--X21--Q41--X22--Q42--
         |    |    |    |    |    |
        Q50--Z20--Q51--Z21--Q52--Z22--
    其中
             Q50
              |
        Q02--X00--Q00
              |
             Q10

             Q42
              |
        Q52--Z22--Q50
              |
             Q02
    '''
    def __init__(self, L):      #toric code数据比特和稳定子的初始化，稳定子也就是对应了测量比特
        '''Toric code of ``2 L**2`` physical qubits and distance ``L``.'''
        self.L = L                                           # toric code的码距
        self.Xflips = np.zeros((2*L,L), dtype=np.dtype('b')) # 发生了 X 错误的qubits qubits where an X error occured 为什么是2L*L，这数对吗；数据类型为byte
        self.Zflips = np.zeros((2*L,L), dtype=np.dtype('b')) # 发生了 Z 错误的qubits qubits where a  Z error occured
        self._Xstab = np.empty((L,L), dtype=np.dtype('b'))   # X稳定子的矩阵表示
        self._Zstab = np.empty((L,L), dtype=np.dtype('b'))   # Z稳定子的矩阵表示

    @property
    def flatXflips2Zstab(self):
        L = self.L
        _flatXflips2Zstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))      #初始化了一个数组用于输出，维度为L^2×2L^2
        for i, j in itertools.product(range(L),range(L)):                      #笛卡尔积，i和j都是从0到L-1
            _flatXflips2Zstab[i*L+j, (2*i  )%(2*L)*L+(j  )%L] = 1              #上
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j  )%L] = 1              #下
            _flatXflips2Zstab[i*L+j, (2*i+2)%(2*L)*L+(j  )%L] = 1              #左
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j+1)%L] = 1              #右
        return _flatXflips2Zstab                                                #得看这个返回值之后干什么用了

    @property
    def flatZflips2Xstab(self):#同上
        L = self.L
        _flatZflips2Xstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))
        for i, j in itertools.product(range(L),range(L)):
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+1)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j  )%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+3)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j+1)%L] = 1
        return _flatZflips2Xstab

    @property
    def flatXflips2Zerr(self):
        L = self.L
        _flatXflips2Zerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatXflips2Zerr[0, (2*k+1)%(2*L)*L+(0  )%L] = 1    #垂直方向的 Z 错误
            _flatXflips2Zerr[1, (2*0  )%(2*L)*L+(k  )%L] = 1    #水平方向的 Z 错误
        return _flatXflips2Zerr

    @property
    def flatZflips2Xerr(self):
        L = self.L
        _flatZflips2Xerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatZflips2Xerr[0, (2*0+1)%(2*L)*L+(k  )%L] = 1    #垂直方向的 X 错误
            _flatZflips2Xerr[1, (2*k  )%(2*L)*L+(0  )%L] = 1    #水平方向的 X 错误
        return _flatZflips2Xerr

    def H(self, Z=True, X=False):
        H = []
        if Z:
            H.append(self.flatXflips2Zstab)
        if X:
            H.append(self.flatZflips2Xstab)
        H = scipy.linalg.block_diag(*H)             #把H中元素都放到对角线上形成个大矩阵返回
        return H

    def E(self, Z=True, X=False):
        E = []
        if Z:
            E.append(self.flatXflips2Zerr)
        if X:
            E.append(self.flatZflips2Xerr)
        E = scipy.linalg.block_diag(*E)
        return E

    def Zstabilizer(self):
        '''Return all measurements of the Z stabilizer with ``true`` marking non-trivial.'''
        #根据错误情况分析稳定子的样子
        #相当于实现了一次测量，输出稳定子应该的样子
        stab = self._Zstab
        X = self.Xflips
        stab[0:-1,0:-1] = X[0:-2:2,0:-1:] ^ X[1:-1:2,0:-1:] ^ X[2::2,0:-1:] ^ X[1:-1:2,1::]
        stab[  -1,0:-1] = X[  -2  ,0:-1:] ^ X[  -1  ,0:-1:] ^ X[  0 ,0:-1:] ^ X[  -1  ,1::]
        stab[0:-1,  -1] = X[0:-2:2,  -1 ] ^ X[1:-1:2,  -1 ] ^ X[2::2,  -1 ] ^ X[1:-1:2,  0]
        stab[  -1,  -1] = X[  -2  ,  -1 ] ^ X[  -1  ,  -1 ] ^ X[  0 ,  -1 ] ^ X[  -1  ,  0]
        return stab

    def Xstabilizer(self):
        '''Return all measurements of the X stabilizer with ``true`` marking non-trivial.'''
        #和Z稳定子
        stab = self._Xstab
        Z = self.Zflips
        stab[1:,1:] = Z[1:-2:2,1:] ^ Z[2:-1:2,0:-1] ^ Z[3::2,1:] ^ Z[2:-1:2,1:]
        stab[0 ,1:] = Z[  -1  ,1:] ^ Z[   0  ,0:-1] ^ Z[  1 ,1:] ^ Z[   0  ,1:]
        stab[1:,0 ] = Z[1:-2:2,0 ] ^ Z[2:-1:2,  -1] ^ Z[3::2,0 ] ^ Z[2:-1:2,0 ]
        stab[0 ,0 ] = Z[  -1  ,0 ] ^ Z[   0  ,  -1] ^ Z[  1 ,0 ] ^ Z[   0  ,0 ]
        return stab

    #绘制 qubit 翻转的坐标
    def _plot_flips(self, s, flips_yx, label):
        '''Given an array of yx coordiante plot qubit flips on subplot ``s``.'''
        if not len(flips_yx): return
        y, x = flips_yx
        x = x.astype(float)
        x[y%2==0] += 0.5
        x = np.concatenate([x, x-self.L, x])
        y = np.concatenate([y/2., y/2., y/2.-self.L])
        s.plot(x, y,'o', ms=50/self.L, label=label)

    def plot(self, legend=True, stabs=True):
        '''Plot the state of the system (including stabilizers).'''
        f = plt.figure(figsize=(5,5))#设置图是几x几
        s = f.add_subplot(1,1,1)#子图
        self._plot_legend = legend

        self._plot_flips(s, self.Xflips.nonzero(), label='X')
        self._plot_flips(s, self.Zflips.nonzero(), label='Z')
        self._plot_flips(s, (self.Xflips & self.Zflips).nonzero(), label='Y')

        if stabs:
            y, x = self.Zstabilizer().nonzero()
            x = np.concatenate([x+0.5, x+0.5-self.L, x+0.5, x+0.5-self.L])
            y = np.concatenate([y+0.5, y+0.5, y+0.5-self.L, y+0.5-self.L])
            # s.plot(x,y,'s', mew=0, ms=95/self.L, label='plaq')#new
            s.plot(x,y,'s', mew=0, ms=190/self.L, label='plaq')

            y, x = self.Xstabilizer().nonzero()
            # s.plot(x, y, '+', mew=50/self.L, ms=100/self.L, label='star')#new
            plt.xlabel('Surface code edge')
            plt.ylabel('Surface code edge')
            # plt.legend(fontsize = 20)
            # plt.grid(axis='y', linestyle='-', linewidth=0.5)
            fig = plt.gcf()
            s.plot(x, y, '+', mew=100/self.L, ms=200/self.L, label='star')

        s.set_xticks(range(0,self.L))
        s.set_yticks(range(0,self.L))
        s.set_xlim(-0.6,self.L-0.4)
        s.set_ylim(-0.6,self.L-0.4)
        s.invert_yaxis()
        # plt.show()
        for tic in s.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        for tic in s.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        s.grid()
        if legend:
            s.legend(markerscale=0.24, loc='upper right')
            # s.legend()
        plt.show()#画图
        return f, s

    #画了一个X/Z稳定子的图，顶点为稳定子，边为计算得来的两个顶点之间的距离
    def _wgraph(self, operator):
        g = nx.Graph()
        if operator == 'Z':
            nodes = zip(*self.Zstabilizer().nonzero())  #返回Z稳定子非平凡的行号和列号
        elif operator == 'X':
            nodes = zip(*self.Xstabilizer().nonzero())
        def dist(node1, node2):
            dy = abs(node1[0]-node2[0])
            dy = min(self.L-dy, dy)
            dx = abs(node1[1]-node2[1])
            dx = min(self.L-dx, dx)
            return dx+dy
        g.f((node1, node2, -dist(node1, node2))
            for node1, node2 in itertools.combinations(nodes, 2))
        return g

    #带有结点间距离的Z稳定子图
    def Zwgraph(self):
        '''The distance graph for non-trivial Z stabilizer.'''
        return self._wgraph('Z')

    def Xwgraph(self):
        '''The distance graph for non-trivial X stabilizer.'''
        return self._wgraph('X')

    def Zcorrections(self):
        '''Qubits on which to apply Z operator to fix the X stabilizer.'''
        L = self.L
        graph = self.Xwgraph()
        matches = {tuple(sorted(_)) for _ in
                   nx.max_weight_matching(graph, maxcardinality=True)}  #实现一个权重的最大匹配
                   # nx.max_weight_matching(graph, maxcardinality=True).items()}  #实现一个权重的最大匹配
        qubits = set()
        for (y1, x1), (y2, x2) in matches:  #计算需要纠正的qubit的位置
            ym, yM = 2*min(y1, y2), 2*max(y1, y2)
            if yM-ym > L:
                ym, yM = yM, ym+2*L
                horizontal = yM if (x2-x1)*(y2-y1)<0 else ym
            else:
                horizontal = ym if (x2-x1)*(y2-y1)<0 else yM
            xm, xM = min(x1, x2), max(x1, x2)
            if xM-xm > L/2:
                xm, xM = xM, xm+L
                vertical = xM
            else:
                vertical = xm
            qubits.update((horizontal%(2*L), _%L) for _ in range(xm, xM))
            qubits.update(((_+1)%(2*L), vertical%L) for _ in range(ym, yM, 2))
        return matches, qubits      #qubits表示需要翻转的量子比特位置的集合

    def Xcorrections(self):
        '''Qubits on which to apply X operator to fix the Z stabilizer.'''
        L = self.L
        graph = self.Zwgraph()
        matches = {tuple(sorted(_)) for _ in
                   nx.max_weight_matching(graph, maxcardinality=True)}
        # print("matches: ", matches)
                   # nx.max_weight_matching(graph, maxcardinality=True).items()}
        qubits = set()
        for (y1, x1), (y2, x2) in matches:
            ym, yM = 2*min(y1, y2), 2*max(y1, y2)
            if yM-ym > L:
                ym, yM = yM, ym+2*L
                horizontal = yM if (x2-x1)*(y2-y1)<0 else ym
            else:
                horizontal = ym if (x2-x1)*(y2-y1)<0 else yM
            xm, xM = min(x1, x2), max(x1, x2)
            if xM-xm > L/2:
                xm, xM = xM, xm+L
                vertical = xM
            else:
                vertical = xm
            qubits.update(((horizontal+1)%(2*L), (_+1)%L) for _ in range(xm, xM))
            qubits.update(((_+2)%(2*L), vertical%L) for _ in range(ym, yM, 2))
        return matches, qubits

    def plot_corrections(self, s, plot_matches=False):
        '''Add to subplot ``s`` the corrections that have to be performed according to min. weight matching.'''
        def stitch_torus(y1, y2):
            if abs(y1-y2)>L/2:
                return (y1+L, y2-L) if y1<y2 else (y1-L, y2+L)
            return y1, y2
        def shorten(y1,y2):
            if y1==y2:
                return y1, y2
            return (y1+0.15, y2-0.15) if y1<y2 else (y1-0.15, y2+0.15)
        S = shorten
        matches, qubits = self.Xcorrections()
        L = self.L
        if matches:
            if plot_matches:
                for ((y1,x1),(y2,x2)) in np.array(list(matches))+0.5:
                    Y1, Y2 = stitch_torus(y1,y2)
                    X1, X2 = stitch_torus(x1,x2)
                    s.plot(S(x1,X2), S(y1,Y2), 'k-', lw=20/self.L)
                    s.plot(S(X1,x2), S(Y1,y2), 'k-', lw=20/self.L)
            y, x = np.array(list(qubits)).T
            cX = np.array([y,x])
        else:
            cX = np.array([[],[]])

        matches, qubits = self.Zcorrections()
        if matches:
            if plot_matches:
                matches = np.array(list(matches))
                for ((y1,x1),(y2,x2)) in matches:
                    Y1, Y2 = stitch_torus(y1,y2)
                    X1, X2 = stitch_torus(x1,x2)
                    s.plot(S(x1,X2), S(y1,Y2), 'k-', lw=20/self.L)
                    s.plot(S(X1,x2), S(Y1,y2), 'k-', lw=20/self.L)
            y, x = np.array(list(qubits)).T
            cZ = np.array([y,x])
        else:
            cZ = np.array([[],[]])
        self._plot_flips(s, cX, label='cX')
        self._plot_flips(s, cZ, label='cZ')
        cY = np.array(list(set(zip(*cZ)).intersection(set(zip(*cX))))).T
        self._plot_flips(s, cY, label='cY')
        if self._plot_legend: s.legend()

    #以去极化错误模型为准，对当前的toric code随机施加错误，细节还是得看
    def add_errors(self, p): #TODO probably faster with numba
        '''Add X, Y, Z errors at rate ``(1-p)/3`` each, e.g. depolarization at ``1-p``.'''
        rand = np.random.rand(self.L*2, self.L)
        q = (1-p)/3                             #错误率均分
        x_flips =                rand<  q       #随机数在哪个范围发生 x 翻转；bool类型矩阵
        z_flips =   (q<=rand) & (rand<2*q)      #bool类型矩阵
        both    = (2*q<=rand) & (rand<3*q)      #Y错误；bool类型矩阵
        self.Xflips ^= x_flips
        self.Xflips ^= both
        #这样的处理相当于both是发生X的概率了，x_flips保持不变时表示的是
        self.Zflips ^= z_flips
        self.Zflips ^= both

    def perform_perfect_correction(self):
        self.Xflips[list(zip(*self.Xcorrections()[1]))] ^= True     #对每个X纠错的位置，将对应的Xfilps值取反
        self.Zflips[list(zip(*self.Zcorrections()[1]))] ^= True

    def logical_errors(self):
        z1 = np.logical_xor.reduce(self.Xflips[1::2,0])
        z2 = np.logical_xor.reduce(self.Xflips[0,:])
        x1 = np.logical_xor.reduce(self.Zflips[1,:])
        x2 = np.logical_xor.reduce(self.Zflips[0::2,0])
        return z1, z2, x1, x2

    def step_error_and_perfect_correction(self, p):
        self.add_errors(p)
        self.perform_perfect_correction()
        return not any(self.logical_errors())

    @staticmethod
    def assert_correctness():
        '''A bunch of functionality is implemented in multiple ways - here we assert they are equivalent.'''
        c = 0
        while c<1000:
            t = ToricCode(3)
            t.add_errors(0.750)
            # Computing stabilizers and measurements with linear algebra and with explicit elementwise ops.
            stabz = t.Zstabilizer().ravel()#Z稳定子的当前表现【因为其中包括Xflips和Zstab的共同作用】
            stabzm = t.flatXflips2Zstab.dot(t.Xflips.ravel()) % 2
            stabx = t.Xstabilizer().ravel()
            stabxm = t.flatZflips2Xstab.dot(t.Zflips.ravel()) % 2
            errz = t.logical_errors()[0:2]
            #print("errz")
            errx = t.logical_errors()[2:4]
            errzm = t.flatXflips2Zerr.dot(t.Xflips.ravel()) % 2
            errxm = t.flatZflips2Xerr.dot(t.Zflips.ravel()) % 2
            assert np.all(stabz==stabzm)
            assert np.all(stabx==stabxm)
            assert np.all(errz==errzm)
            assert np.all(errx==errxm)
            c += 1
            if not c%100:
                print('\r',c,end='',flush=True)
            # t.plot()#new
        print('\n')


def sample(L, p, samples=1000, cutoff=200):
    '''Repeated single shot corrections for the toric code with perfect measurements.

    Return an array of nb of cycles until failure for a given L and p.'''
    results = []
    for _ in trange(samples, desc='%d; %.2f'%(L,p), leave=False):#一个统计的进度条
        code = ToricCode(L)
        i = 1
        while code.step_error_and_perfect_correction(p) and i<cutoff:#cutoff次，有多少次成功的纠错
            i+=1
        # print("cutoff: ", cutoff, " i = ", i, " samples = ", samples);
        results.append(i)
    return np.array(results, dtype=int)

def stat_estimator(samples, cutoff=200, confidence=0.99):
    '''Max Likelihood Estimator for censored exponential distribution.

    See "Estimation of Parameters of Truncated or Censored Exponential Distributions",
    Walter L. Deemer and David F. Votaw'''
    samples = samples.astype(float)
    n = (samples<cutoff).sum()
    N = len(samples)
    estimate = n/samples.sum()
    y_conf = stats.norm.ppf((1+confidence)/2)
    y = lambda c: N**0.5*(estimate/c-1)*(1-np.exp(-c*cutoff))**0.5
    low  = optimize.root(lambda c: y(c)-y_conf, estimate)
    high = optimize.root(lambda c: y(c)+y_conf, estimate)
    if not (low.success and high.success):
        raise RuntimeError('Could not find confidence interval for the given samples!')
    return np.array([1/estimate, 1/high.x, 1/low.x])

def find_threshold(Lsmall=3, Llarge=5, p=0.8, high=1, low=0.79, samples=1000, logfile=None):#logfile:是否写入日志，不写就画图输出
    '''Use binary search (between two sizes of codes) to find the threshold for the toric code.'''
    print("find_threshold start.")
    ps = []#p的列表
    samples_small = []
    samples_large = []
    def step(p):
        print(" into step p: ", p);
        ps.append(p)
        samples_small.append(stat_estimator(sample(Lsmall, p, samples=samples)))
        samples_large.append(stat_estimator(sample(Llarge, p, samples=samples)))
    def intersection(xs, y1s, y2s, log=True):
        d = np.linalg.det
        if log:
            y1s, y2s = np.log([y1s,y2s])
        ones = np.array([1.,1.])
        dx  = d([xs , ones])
        dy1 = d([y1s, ones])
        dy2 = d([y2s, ones])
        x = (d([xs, y1s])-d([xs, y2s])) / (dy2-dy1)
        y = (d([xs, y1s])*dy2 - d([xs, y2s])*dy1) / dx / (dy2-dy1)
        if log:
            y = np.exp(y)
        # print("  a intersection: xs = ", xs, " y1s = ", y1s, " y2s = ", y2s, " return x = ", x, " y = ", y)
        return x, y

    count = 1;
    print("Count: ", count)
    step(p)
    if logfile:#写入日志文件
        with open(logfile, 'w') as f:
            print("write into logfile at first");
            ss = samples_small[0]
            sl = samples_large[0]
            f.write(str((np.vstack([ps, [ss[0]], [ss[1]-ss[0]], [ss[2]-ss[0]], [sl[0]], [sl[1]-sl[0]], [sl[2]-sl[0]]]), (ss[0]+sl[0])/2, ps[0])))
    else:
        print("draw picture")
        f = plt.figure()
        s = f.add_subplot(1,1,1)
        # display.display(f)
    print(" samples_small[-1][0]: ", samples_small[-1][0], " sample_small[-1][1]: ", samples_small[-1][1], " samples_small[-1][2]: ", samples_small[-1][2])
    print(" samples_large[-1][0]: ", samples_large[-1][0], " samples_large[-1][1]: ", samples_large[-1][1], " samples_large[-1][2]: ", samples_large[-1][2])
    print("ready to while")
    while not (samples_large[-1][1]<samples_small[-1][0]<samples_large[-1][2]#0:1/estimate, 1: 1/high.x, 2: 1/low.x
            or samples_small[-1][1]<samples_large[-1][0]<samples_small[-1][2]):#直到两个曲线重叠？
        count += 1
        print("Count: ", count)
        print(" samples_small[-1][0]: ", samples_small[-1][0], " sample_small[-1][1]: ", samples_small[-1][1], " samples_small[-1][2]: ", samples_small[-1][2])
        print(" samples_large[-1][0]: ", samples_large[-1][0], " samples_large[-1][1]: ", samples_large[-1][1], " samples_large[-1][2]: ", samples_large[-1][2])
        if samples_small[-1][0]<samples_large[-1][0]:#小尺寸小于大尺寸
            print(" bs: p become low")
            p, high = low+(ps[-1]-low)/2, p
        else:
            print(" bs: p become high")
            p, low = ps[-1]+(high-ps[-1])/2, p
        step(p)
        _argsort = np.argsort(ps)
        _ps = np.array(ps)[_argsort]#给ps排序，为后面画图准备
        _ss = np.array(samples_small)
        _small = _ss[_argsort,0]
        _small_err = np.abs(_ss[_argsort,1:].T - _small)
        _sl = np.array(samples_large)
        _large = _sl[_argsort,0]
        _large_err = np.abs(_sl[_argsort,1:].T - _large)
        ix, iy = intersection(ps[-2:],[_[0] for _ in samples_small[-2:]],[_[0] for _ in samples_large[-2:]])
        print("ix = ", ix, " iy = ", iy)
        if logfile:#写入日志文件
            with open(logfile, 'w') as f:
                print(" write into logfile : ", logfile);
                f.write(str((np.vstack([_ps, _small, _small_err, _large, _large_err]), iy, ix)))
        else:#画图可视化输出
            print(" plot output")
            s.clear()
            s.errorbar(_ps,_small,yerr=_small_err,alpha=0.6,label=str(Lsmall))
            s.errorbar(_ps,_large,yerr=_large_err,alpha=0.6,label=str(Llarge))
            s.plot([ix],[iy],'ro',alpha=0.5)
            s.set_title('intersection at p = %f'%ix)
            s.set_yscale('log')
            display.clear_output(wait=True)
            display.display(f)
            plt.show()#new
    print("find_threshold end.")
    return ps, samples_small, samples_large


def generate_training_data(l=3, p=0.9, train_size=2000000, test_size=100000): # TODO duplicated code with data_generator in neural.py
    '''Generate errors and corresponding stabilizers at a given `p` for the toric code.

    The samples with no errors are skipped.
    It counts and prints out how many of the errors are fixed by MWPM.

    returns: (Zstab_x_train, Zstab_y_train, Xstab_x_train, Xstab_y_train,
              Zstab_x_test,  Zstab_y_test,  Xstab_x_test,  Xstab_y_test)
    train_size: 训练集大小
    test_size: 测试集大小
    '''
    Zstab_x_train = np.zeros((train_size, l**2))
    Zstab_y_train = np.zeros((train_size, 2*l**2))
    Xstab_x_train = np.zeros((train_size, l**2))
    Xstab_y_train = np.zeros((train_size, 2*l**2))
    for i in trange(train_size):
        t = ToricCode(l)
        t.add_errors(p)
        while not (np.any(t.Xflips) or np.any(t.Zflips)):
            t = ToricCode(l)
            t.add_errors(p)
        Zstab_x_train[i,:] = t.Zstabilizer().ravel()
        Zstab_y_train[i,:] = t.Xflips.ravel()
        Xstab_x_train[i,:] = t.Xstabilizer().ravel()
        Xstab_y_train[i,:] = t.Zflips.ravel()
    Zstab_x_test = np.zeros((test_size, l**2))
    Zstab_y_test = np.zeros((test_size, 2*l**2))
    Xstab_x_test = np.zeros((test_size, l**2))
    Xstab_y_test = np.zeros((test_size, 2*l**2))
    errors = xstab_errors = zstab_errors = 0
    for i in trange(test_size):
        t = ToricCode(l)
        t.add_errors(p)
        while not (np.any(t.Xflips) or np.any(t.Zflips)):
            t = ToricCode(l)
            t.add_errors(p)
        Zstab_x_test[i,:] = t.Zstabilizer().ravel()
        Zstab_y_test[i,:] = t.Xflips.ravel()
        Xstab_x_test[i,:] = t.Xstabilizer().ravel()
        Xstab_y_test[i,:] = t.Zflips.ravel()
        t.perform_perfect_correction()
        errors += any(t.logical_errors())
        xstab_errors += any(t.logical_errors()[0:2])
        zstab_errors += any(t.logical_errors()[2:4])
    decoded_fraction = 1 - errors/test_size
    xstab_decoded_fraction = 1 - xstab_errors/test_size
    zstab_decoded_fraction = 1 - zstab_errors/test_size
    print('decoded_fraction, zstab_decoded_fraction, xstab_decoded_fraction =')
    print(decoded_fraction, zstab_decoded_fraction, xstab_decoded_fraction)
    return ((Zstab_x_train, Zstab_y_train, Xstab_x_train, Xstab_y_train,
             Zstab_x_test,  Zstab_y_test,  Xstab_x_test,  Xstab_y_test),
            (decoded_fraction, zstab_decoded_fraction, xstab_decoded_fraction))

# def

# print("assert2.0:")
# ToricCode.assert_correctness()#验证，代码逻辑测试通过
toric = ToricCode(7)
# print(sample(3, 0.9))
toric.add_errors(0.9)
# print("toric code的情况")
print(" Xflips")
print(toric.Xflips)
print(" Zflips")
print(toric.Zflips)
# print(" ")
# # print(toric.)
toric.plot()
# print("flatXflips2Zstab")
# print(toric.flatXflips2Zstab)
# print("flatXflips2Xstab")
# print(toric.flatZflips2Xstab)
# print("flatXflips2Zerr")
# print(toric.flatXflips2Zerr)
# print("flatZflips2Xerr")
# print(toric.flatZflips2Xerr)
# plt.show()
print("program over.")