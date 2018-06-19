from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.
  两层全连接神经网络。这个网络有一个维度为N的输入，一个维度为H的隐层，并且在C个类别上执行分类。
  我们用一个softmax损失函数和在权重矩阵上使用L2正则化来训练网络。这个网络在第一层全连接后面
  使用一个ReLU非线性分类器

  In other words, the network has the following architecture:
  换句话说，这个网络有以下的架构：

  input - fully connected layer - ReLU - fully connected layer - softmax
  输入层 - 全连接层 - ReLu - 全连接层 - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  第二个全连接层的输出是每个分类的得分。
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:
    
    初始化模型。权重使用小的随机数初始化，偏置初始化为0。权重和偏置被存储在变量self.params中，
    这是一个有以下key的字典：

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)
    
    

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    
    输入:
    - 输入大小: 输入数据的维度为 D
    - 隐藏层大小: 在隐藏层中神经元的数量 H
    - 输出层大小: 分类的数目 C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.
    
    为两层全连接神经网络计算损失值和梯度

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.
    
    输入:
    - X: 输入数据的维度 (N, D). 每个 X[i] 是一个训练样本.
    - y: 训练标签的向量。 y[i] 是 X[i] 的标签, 并且每个y[i]是一个在0 <= y[i] < C之间的整数.
      这个参数是可选的，如果没有传我们返回得分数，如果传了我们返回损失分数和梯度。
    - reg: 正则化强度

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].
    
    返回值:
    如果y是空的，返回一个(N, C)维的得分矩阵，其中scores[i, c]是类别c在输入X[i]上面的得分。

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
      
    如果y是空的，则返回一个元组:
    - loss: 这个批次训练样本的损失（数据损失和正则化损失）
    - grads: D字典映射参数名称到损失函数中这些参数的梯度；作为self.params拥有相同的keys
     
    """
    # 从字典中取出变量
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # 前向传播计算
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    # 执行前向传播，为输入计算分类得分。将结果存储在scores变量中，这是一个(N,C)维的数组
    #############################################################################
    pass
    z1 = X.dot(W1) + b1
    a1 = np.maximum(0,z1)
    scores = a1.dot(W2) + b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # 如果目标标签没有给，那么直接结束。返回得分。
    if y is None:
      return scores

    # 计算损失
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    # 完成前向传播，计算损失。其中包括数据损失和W1和W2的正则化损失。将结果存储在loss变量中，
    # 这是一个标量。使用softmax分类器损失。
    #############################################################################
    pass
    f = scores -np.max(scores,axis =1,keepdims =True)
    loss = -f[range(N),y].sum() + np.log(np.exp(f).sum(axis =1)).sum()
    loss = loss /N +0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))
    
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # 反向传播：计算梯度
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    # 计算反向传播，计算权重和偏置的倒数，将结果存储在grads字典中。例如，grads['W1']应该
    # 存储W1上的梯度，并且是一个相同大小的矩阵。
    #############################################################################
    pass
    # 计算分值的梯度
    dscores = probs
    dscores[range(N),y] -= 1
    dscores /= N
    
    # W2和b2
    grads['W2'] = np.dot(a1.T,dscores)
    grads['b2'] = np.sum(dscores,axis=0)
    
    # 反向传播里面的第二个隐藏层
    dhidden = np.dot(dscores,W2.T)
    
    # 激活函数ReLu的梯度
    dhidden[a1<=0] = 0
    
    #关于W1和b1的梯度
    grads['W1']=np.dot(X.T,dhidden)
    grads['b1']=np.sum(dhidden,axis=0)
    
    #加上正则化梯度的部分
    grads['W2']+=reg*W2
    grads['W1']+=reg*W1
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.
    使用随机梯度下降来训练这个神经网络

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    
    输入:
    - X: 一个装有(N,D)维训练数据的numpy数组
    - y: 一个装有(N,1)维训练标签的numpy数组；y[i] = c意味着X[i]有一个标签c，范围为0<=c<C。
    - X_val: 一个装有(N_val,D)维验证数据的numpy数组
    - y_val: 一个装有(N_val,)维验证标签的numpy数组
    - learning_rate: 为了优化而存在的学习率，是一个标量
    - learning_rate_decay: 用于在每个周期后使学习率衰减的一个标量因子
    - reg: 正则化强度标量
    - num_iters: 优化时使用的步骤数
    - batch_size: 每一步长使用的训练样本数
    - verbose: boolean; 如果为true则在优化时打印过程
    """
    
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # 使用SGD来优化self.model中的参数
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      # 创建一个随机的小批量的训练数据和标签，分别存储在X_batch 和y_batch中
      #########################################################################
      pass
      sample_indices = np.random.choice(np.arange(num_train),batch_size)
      X_batch = X[sample_indices]
      y_batch = y[sample_indices]
      
      
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # 使用当前的小批量数据计算损失和梯度
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      # 使用在grads字典中的梯度来更新使用随机梯度下降的网络中的参数（存储在字典中的self.params）
      # 你将会需要使用存储在上面定义的grads字典中的梯度
      #########################################################################
      pass
      self.params['W1'] += -learning_rate * grads['W1']
      self.params['b1'] += -learning_rate * grads['b1']
      self.params['W2'] += -learning_rate * grads['W2']
      self.params['b2'] += -learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # 每个周期，检查训练和验证的准确率并且衰减学习率
      if it % iterations_per_epoch == 0:
        # 检查准确率
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # 衰减学习率
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.
    
    使用两层网络的训练好的权重来预测数据点的标签。为每个数据点我们为每个类别C预测分数，
    并且分配每个数据点到得分最高的分类中去

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
      
    输入:
    - X: 一个(N,D)维的numpy数组有N个D维数据点来进行分类

    返回值:
    - y_pred: 一个(N,)的numpy数组给出每个X中的元素的预测标签。对于所有的i，y_pred[i]=c意味着X[i]被预测为类别c，其中c的大小为0<=c<C
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    z1 = X.dot(self.params['W1'])+self.params['b1']
    a1 = np.maximum(0,z1) #通过ReLU激活函数
    scores = a1.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores,axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


