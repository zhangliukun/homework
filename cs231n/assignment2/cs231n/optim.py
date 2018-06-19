import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

这个文件实现了很多通用的训练神经网络的一阶更新规则，每个更新规则接收当前的权重和这些权重的损失的梯度，
然后生成下一个权重集。每个更新规则有相同的接口：

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights. 当前的权重矩阵。
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.  和w形状相同w的损失梯度。
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.  一个包含超参数的字典，
    如果更新规则要求在许多迭代中缓存值，这样config将会存储这些缓存值。

Returns:
  - next_w: The next point after the update. 更新过后下一个点
  - config: The config dictionary to be passed to the next iteration of the
    update rule.  更新规则的下一次迭代传的config字典

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

注意：对于大多数的更新规则来说，默认的学习率并不表现的很好；但是默认的值其它超参数对于很多不同的
问题表现的不错。

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.

为了效率，更新规则将会执行就地更新，转变w且将next_w的值设置为w
"""


def sgd(w, dw, config=None):
    """
    执行vanilla随机梯度下降

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2) #键不存在时，设置学习率

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    执行动量随机梯度下降

    config format:
    - learning_rate: 学习率标量
    - momentum: 动量是一个值为0到1的标量，如果不使用则设为0
    - velocity: 一个和w与dw形状相同的数组，被用来存储一个梯度的平均变化值
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    # 实现动量更新公式。将更新的值存储在变量next_w中，你也应该使用和更新速度v
    ###########################################################################
    pass
    v = config['momentum']*v - config['learning_rate']*dw
    next_w = w + v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v

    return next_w, config



def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    使用RMSProp更新规则，它使用一个梯度平方的滑动平均平均数来设置自适应的每个参数的学习率

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.  梯度缓存的平方的衰退率
    - epsilon: 一个小数防止除数为0
    - cache: Moving average of second moments of gradients.梯度第二矩的移动平均
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x #
    # in the next_x variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    pass
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * (dx**2)
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    使用Adam更新规则，它将梯度和梯度的平方的滑动平均值与一个偏置校准项合并起来使用。

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.梯度一次阶滑动平均的退火率
    - beta2: Decay rate for moving average of second moment of gradient.梯度二次阶滑动平均的退火率
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient. 梯度滑动平均
    - v: Moving average of squared gradient.  梯度平方的滑动平均
    - t: Iteration number. 迭代次数
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in #
    # the next_x variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    ###########################################################################
    pass
    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx**2)
    mb = config['m'] / (1 - config['beta1']**config['t'])
    vb = config['v'] / (1 - config['beta2']**config['t'])
    next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['epsilon'])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config
