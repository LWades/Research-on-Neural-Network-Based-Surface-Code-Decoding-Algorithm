import codes
from codes import ToricCode
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Nadam
# from keras import binary_crossentropy
# from keras.objectives import binary_crossentropy
# from keras.layers import BatchNormalization
from keras.layers import BatchNormalization
# from keras.layers.normalization import BatchNormalization
import tensorflow as tf

# import tensorflow as tf

F = lambda _: K.cast(_,
                     'float32')  # TODO XXX there must be a better way to calculate mean than this cast-first approach


class CodeCosts:
    def __init__(self, L, code, Z, X, normcentererr_p=None):
        if normcentererr_p:  # 代码中的注释提示这种方法可能是不恰当的，特别是在使用二元交叉熵作为损失函数的情况下。二元交叉熵通常用于分类任务，它测量实际输出和预测输出之间的差异。在这种情况下，标准化和中心化误差可能不适用或不必要，因为二元交叉熵已经是一个处理 0 和 1 输出的标准方法。此外，标准化和中心化可能会改变原始数据的特性，这在某些情况下可能不利于模型的性能。
            raise NotImplementedError(
                'Throughout the entire codebase, the normalization and centering of the error, might be wrong... Or to be more precise, it might just be plain stupid, given that we are using binary crossentropy as loss.')
        self.L = L
        code = code(L)
        H = code.H(Z, X)
        E = code.E(Z, X)
        # 将 H 转换为 Keras 变量
        self.H = K.variable(value=H)  # TODO should be sparse
        # 将 E 转换为 Keras 变量
        self.E = K.variable(value=E)  # TODO should be sparse
        self.p = normcentererr_p

    def exact_reversal(self, y_true, y_pred):  # y_true 为真实值；y_pred 为预测值
        """
        Fraction exactly predicted qubit flips.
        exact_reversal 方法计算了模型在预测量子比特翻转方面的精确性能，如果提供了标准化和中心化参数，则在计算之前对数据进行还原。
        这种方法在评估模型预测量子态翻转的准确性时可能特别有用。
        应该是一个评价性能的方法，在原代码中没有被调用
        """
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        return K.mean(F(K.all(K.equal(y_true, K.round(y_pred)), axis=-1)))

    def non_triv_stab_expanded(self, y_true, y_pred):
        """
        Whether the stabilizer after correction is not trivial.
        似乎用于判断在应用预测的量子比特翻转后，量子系统的稳定子是否非平凡。这在评估量子纠错算法的有效性时可能非常重要，特别是在判断是否正确地识别和纠正了错误时。
        """
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        return K.any(K.dot(self.H, K.transpose((K.round(y_pred) + y_true) % 2)) % 2, axis=0)

    def logic_error_expanded(self, y_true, y_pred):
        """
        Whether there is a logical error after correction.
        目的是在量子纠错的上下文中判断在应用纠正操作之后是否存在逻辑错误。这是一个关键的功能，因为在量子计算中，识别和纠正逻辑错误是确保信息准确性的重要部分
        """
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        return K.any(K.dot(self.E, K.transpose((K.round(y_pred) + y_true) % 2)) % 2, axis=0)

    def triv_stab(self, y_true, y_pred):
        """
        Fraction trivial stabilizer after corrections.
        算纠正后稳定子为平凡（trivial）的比例。这个方法是 non_triv_stab_expanded 方法的补充，
        后者用于判断纠正后的稳定子是否非平凡（non-trivial）。
        在量子纠错代码中，了解稳定子的状态（是否为平凡或非平凡）对于评估纠错性能是重要的。
        """
        return 1 - K.mean(F(self.non_triv_stab_expanded(y_true, y_pred)))

    def no_error(self, y_true, y_pred):
        """
        Fraction no logical errors after corrections.
        用于计算在纠正操作后没有逻辑错误的比例。这个方法是 logic_error_expanded 方法的补充，
        后者用于判断在纠正操作之后是否存在逻辑错误。
        在量子纠错代码中，评估算法能够有效避免逻辑错误的能力是非常重要的。让我们详细解析这个方法
        """
        return 1 - K.mean(F(self.logic_error_expanded(y_true, y_pred)))

    def triv_no_error(self, y_true, y_pred):
        """
        Fraction with trivial stabilizer and no error.
        目的是计算在量子纠错过程中同时满足两个条件的比例：稳定子是平凡的（trivial），并且没有逻辑错误。
        这是对先前定义的 non_triv_stab_expanded 和 logic_error_expanded 方法的综合应用，反映了量子纠错算法的整体效能。
        """
        # TODO XXX Those casts (the F function) should not be there! This should be logical operations
        triv_stab = 1 - F(self.non_triv_stab_expanded(y_true, y_pred))
        no_err = 1 - F(self.logic_error_expanded(y_true, y_pred))
        return K.mean(no_err * triv_stab)

    def e_binary_crossentropy(self, y_true, y_pred):
        """
        定义了一种计算二元交叉熵损失的方式，适用于评估量子纠错算法中预测结果的准确性。
        二元交叉熵是一种常见的损失函数，用于度量两个概率分布之间的差异，特别是在二分类问题中。
        """
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

    def s_binary_crossentropy(self, y_true, y_pred):
        """
        定义了一种特定的二元交叉熵损失计算方式，适用于评估量子纠错算法中稳定子的预测准确性。
        这个方法的特殊之处在于它不直接计算原始预测值 (y_pred) 和真实值 (y_true) 之间的二元交叉熵，
        而是首先通过一系列变换计算出稳定子的预测值和真实值，然后再计算这些稳定子值之间的二元交叉熵
        """
        if self.p:
            y_pred = undo_normcentererr(y_pred, self.p)
            y_true = undo_normcentererr(y_true, self.p)
        s_true = K.dot(y_true, K.transpose(self.H)) % 2
        twopminusone = 2 * y_pred - 1
        s_pred = (1 - tf.math.real(
            K.exp(K.dot(K.log(tf.cast(twopminusone, tf.complex64)), tf.cast(K.transpose(self.H), tf.complex64))))) / 2
        # s_pred = ( 1 - tf.real(K.exp(K.dot(K.log(tf.cast(twopminusone, tf.complex64)), tf.cast(K.transpose(self.H), tf.complex64)))) ) / 2
        return K.mean(K.binary_crossentropy(s_pred, s_true), axis=-1)

    def se_binary_crossentropy(self, y_true, y_pred):
        """
        定义了一个组合的损失函数，它将 e_binary_crossentropy（针对元素的二元交叉熵损失）和 s_binary_crossentropy（针对稳定子的二元交叉熵损失）结合起来，
        以评估量子纠错算法的整体性能。这种组合方法可以提供一个更全面的性能指标，因为它同时考虑了元素级别和稳定子级别的准确性
        """
        return 2. / 3. * self.e_binary_crossentropy(y_true, y_pred) + 1. / 3. * self.s_binary_crossentropy(y_true,
                                                                                                           y_pred)


def create_model(L, hidden_sizes=[4], hidden_act='tanh', act='sigmoid', loss='binary_crossentropy',
                 Z=True, X=False, learning_rate=0.002,
                 normcentererr_p=None, batchnorm=0):
    """
    构建一个神经网络模型
    :param L: 码距
    :param hidden_sizes: 隐藏层大小「数组形式」
    :param hidden_act: 隐藏层的激活函数；默认'tanh'
    :param act: 输出层的激活函数；默认'sigmoid'
    :param loss: 损失函数的名称 或者 自定义损失函数；默认'binary_crossentropy'
    :param Z: 要不要训练 Z 稳定子；要不要纠正 X 错误
    :param X: 要不要训练 X 稳定子；要不要纠正 Z 错误
    :param learning_rate: 优化器的学习率
    :param normcentererr_p: 用于标准化和中心化误差的参数
    :param batchnorm: 批量归一化的动量参数「不懂」
    :return:
    """
    in_dim = L ** 2 * (X + Z)  # 输入层维度
    out_dim = 2 * L ** 2 * (X + Z)  # 输出层维度
    model = Sequential()  # 使用 Keras 的 Sequential 模型
    model.add(
        Dense(int(hidden_sizes[0] * out_dim), input_dim=in_dim, kernel_initializer='glorot_uniform'))  # 增加 Dense（全连接）层，并设定 Glorot 均匀初始化这样的权重初始化方法
    if batchnorm:   # 每个全连接层之后加不加批量归一化层
        model.add(BatchNormalization(momentum=batchnorm))
    model.add(Activation(hidden_act))   # 给模型加隐藏层激活函数
    for s in hidden_sizes[1:]:          # 加上每一个隐藏层
        model.add(Dense(int(s * out_dim), kernel_initializer='glorot_uniform'))
        if batchnorm:
            model.add(BatchNormalization(momentum=batchnorm))
        model.add(Activation(hidden_act))
    model.add(Dense(out_dim, kernel_initializer='glorot_uniform'))
    if batchnorm:
        model.add(BatchNormalization(momentum=batchnorm))
    model.add(Activation(act))      # 最后设置上输出层的激活函数
    c = CodeCosts(L, ToricCode, Z, X, normcentererr_p)              # 设置自定义损失函数
    losses = {'e_binary_crossentropy': c.e_binary_crossentropy,
              's_binary_crossentropy': c.s_binary_crossentropy,
              'se_binary_crossentropy': c.se_binary_crossentropy}
    # 配置神经网络的训练过程
    model.compile(loss=losses.get(loss, loss),
                  optimizer=Nadam(lr=learning_rate),    # 设定了一个优化器
                  metrics=[c.triv_no_error, c.e_binary_crossentropy, c.s_binary_crossentropy]   # 评估指标不会用来训练过程中的权重调整，但提供了模型性能的额外信息，估计是训练的时候输出用的
                  )
    return model


def makeflips(q, out_dimZ, out_dimX):
    """
    用来生成错误，和之前的add_errors差不多吧
    :param q:
    :param out_dimZ:
    :param out_dimX:
    :return:
    """
    flips = np.zeros((out_dimZ + out_dimX,), dtype=np.dtype('b'))   # 初始化
    rand = np.random.rand(
        out_dimZ or out_dimX)  # if neither is zero they have to necessarily be the same (equal to the number of physical qubits)
    both_flips = (2 * q <= rand) & (rand < 3 * q)
    if out_dimZ:  # non-trivial Z stabilizer is caused by flips in the X basis
        x_flips = rand < q
        flips[:out_dimZ] ^= x_flips
        flips[:out_dimZ] ^= both_flips
    if out_dimX:  # non-trivial X stabilizer is caused by flips in the Z basis
        z_flips = (q <= rand) & (rand < 2 * q)
        flips[out_dimZ:out_dimZ + out_dimX] ^= z_flips
        flips[out_dimZ:out_dimZ + out_dimX] ^= both_flips
    return flips


def nonzeroflips(q, out_dimZ, out_dimX):
    """
    用一个while循环来保证这里至少有一个错误
    在现实的量子计算中，完全没有错误的情况是非常罕见的。这个函数通过确保至少有一个翻转发生，提供了更加现实的错误模拟
    :param q:
    :param out_dimZ:
    :param out_dimX:
    :return:
    """
    flips = makeflips(q, out_dimZ, out_dimX)
    while not np.any(flips):
        flips = makeflips(q, out_dimZ, out_dimX)
    return flips


# def data_generator(H, out_dimZ, out_dimX, in_dim, p, batch_size=512, size=None,
#                    normcenterstab=False, normcentererr=False):
#     c = 0
#     q = (1 - p) / 3
#     while True:
#         flips = np.empty((batch_size, out_dimZ + out_dimX), dtype=int)  # TODO dtype? byte?
#         for i in range(batch_size):
#             flips[i, :] = nonzeroflips(q, out_dimZ, out_dimX)  # 这里应该是最终的
#         stabs = np.dot(flips, H.T) % 2  # 使用校验矩阵生成错误症状
#         if normcenterstab:
#             stabs = do_normcenterstab(stabs, p)  # 这里是错误症状结果吧，预处理的是这里
#         if normcentererr:
#             flips = do_normcentererr(flips, p)
#         yield (stabs, flips)  # {错误症状，错误信息}对儿，训练数据，如果之类错误症状不对劲，那就不生成这一个
#         c += 1
#         # if size and c==size:
#         #     raise StopIteration

def data_generator(H, out_dimZ, out_dimX, in_dim, p, batch_size=512, size=None,
                   normcenterstab=False, normcentererr=False):
    c = 0
    q = (1 - p) / 3
    while True:
        flips = np.empty((batch_size, out_dimZ + out_dimX), dtype=int)  # TODO dtype? byte?
        for i in range(batch_size):
            flip = nonzeroflips(q, out_dimZ, out_dimX)
            stab = np.dot(flip, H.T) % 2
            # if i < 10:
            #     print("stab: ")
            #     print(stab)
            g_c = 0
            while codes.symmetry_filter(stab, True, True) and g_c < 10000:
                # print(" make filter")
                # print()
                flip = nonzeroflips(q, out_dimZ, out_dimX)
                stab = np.dot(flip, H.T) % 2
                g_c += 1
            if g_c >= 10000:
                print("sym so much!!!!!")
            flips[i, :] = flip
        stabs = np.dot(flips, H.T) % 2  # 使用校验矩阵生成错误症状
        if normcenterstab:
            stabs = do_normcenterstab(stabs, p)  # 这里是错误症状结果吧，预处理的是这里
        if normcentererr:
            flips = do_normcentererr(flips, p)
        yield (stabs, flips)  # {错误症状，错误信息}对儿，训练数据，如果之类错误症状不对劲，那就不生成这一个
        c += 1
        # if size and c==size:
        #     raise StopIteration

def do_normcenterstab(stabs, p):
    avg = (1 - p) * 2 / 3
    avg_stab = 4 * avg * (1 - avg) ** 3 + 4 * avg ** 3 * (1 - avg)
    var_stab = avg_stab - avg_stab ** 2
    return (stabs - avg_stab) / var_stab ** 0.5


def undo_normcenterstab(stabs, p):
    avg = (1 - p) * 2 / 3
    avg_stab = 4 * avg * (1 - avg) ** 3 + 4 * avg ** 3 * (1 - avg)
    var_stab = avg_stab - avg_stab ** 2
    return stabs * var_stab ** 0.5 + avg_stab


def do_normcentererr(flips, p):
    avg = (1 - p) * 2 / 3
    var = avg - avg ** 2
    return (flips - avg) / var ** 0.5


def undo_normcentererr(flips, p):
    avg = (1 - p) * 2 / 3
    var = avg - avg ** 2
    return flips * var ** 0.5 + avg


def smart_sample(H, stab, pred, sample, giveup):
    '''Sample `pred` until `H@sample==stab`.

    `sample` is modified in place. `giveup` attempts are done at most.
    Returns the number of attempts.'''
    npany = np.any
    nprandomuniform = np.random.uniform
    npsum = np.sum
    npdot = np.dot
    attempts = 1
    mismatch = stab != npdot(H, sample) % 2
    while npany(mismatch) and attempts < giveup:
        propagated = npany(H[mismatch, :], axis=0)
        sample[propagated] = pred[propagated] > nprandomuniform(size=npsum(propagated))
        mismatch = stab != npdot(H, sample) % 2
        attempts += 1
    return attempts
