from __future__ import print_function, division
from builtins import range
import numpy as np


"""
这个文件中定义了循环神经网络通用的层类型
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    
    为一个使用tanh激活函数的vanilla的RNN运行单个时间步长的前向传播。
    
    这个输入数据的维度为D，隐藏状态的维度为H，我们使用小批量的大小为N。

    Inputs:
    - x: 这个时间步长的输入数据, of shape (N, D).
    - prev_h: 上一个时间步长的隐藏状态, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections输入到隐藏连接的权重矩阵, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections隐藏到隐藏的权重矩阵, of shape (H, H)
    - b: Biases of shape (H,)偏置

    Returns a tuple of:
    - next_h: 下一个隐藏层的状态, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    #s_t=f(U x_t+W s_{t-1})
    pass
    a = prev_h.dot(Wh) + x.dot(Wx) + b
    next_h = np.tanh(a)
    cache = (x, prev_h, Wh, Wx, b,next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    vanilla RNN 的单个时间步长的反向传播

    Inputs:
    - dnext_h: 下一个隐藏层的梯度损失
    - cache: 前向传播的Cache对象

    Returns a tuple of:
    - dx: 输入数据的梯度, of shape (N, D)
    - dprev_h: 前一个隐藏层状态的梯度, of shape (N, H)
    - dWx: 输入到隐藏层权重的梯度, of shape (D, H)
    - dWh: 隐藏到隐藏层的权重的梯度, of shape (H, H)
    - db: 偏置向量的梯度, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    # 提示：对于tanh函数，你可以根据tanh的输出值计算局部导数
    ##############################################################################
    pass
    #a = prev_h.dot(Wh) + x.dot(Wx) + b
    #next_h = np.tanh(a)
    x, prev_h, Wh, Wx, b,next_h = cache
    da = dnext_h * (1-next_h*next_h)
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    db = np.sum(da,axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    
    在一整个序列数据上面运行vanilla RNN的前向传播。我们假设输入序列是由T个向量组成，每个维度为D。这个RNN
    使用了一个大小为H的隐藏层，且我们在包含N个序列的小批量数据上运行。在跑完RNN的前向传播以后，我们将所有
    的时间步长的隐藏层状态返回。

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: 初始隐藏层的状态, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    pass
    N,T,D = x.shape
    (H,) = b.shape
    h = np.zeros((N,T,H))
    prev_h = h0
    for t in range(T):
        xt = x[:,t,:]
        next_h,_ = rnn_step_forward(xt,prev_h,Wx,Wh,b)
        prev_h = next_h
        h[:,t,:]= prev_h
        
    cache = (x,h0,Wh,Wx,b,h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    pass
    x, h0, Wh, Wx, b, h = cache
    N, T, H = dh.shape
    _, _, D = x.shape

    next_h = h[:,T-1,:]
    
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx= np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))
    
    for t in range(T):
        t = T-1-t
        xt = x[:,t,:]
    
        if t ==0:
            prev_h = h0
        else:
            prev_h = h[:,t-1,:]
        
        step_cache = (xt, prev_h, Wh, Wx, b, next_h)
        next_h = prev_h
        dnext_h = dh[:,t,:] + dprev_h
        dx[:,t,:], dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_h, step_cache)
        dWx, dWh, db = dWx+dWxt, dWh+dWht, db+dbt
    
    dh0 = dprev_h
        
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    
    单词嵌入的前向传播。我们在每个序列有长度为T的大小为N的小批量数据上才做。我们假设一个词汇表有V个单词，
    指定每一个单词是一个维数为D的向量。

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    pass
    out = W[x,:]
    cache = (W,x)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    
    word embedding的反向传播。我们不能反向传播变成单词因为它们是数字，所以我们只要返回word embedding
    矩阵的梯度就可以了。

    提示: 查看函数 np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    pass
    W, x = cache
    dW = np.zeros_like(W)
    #dW[x] += dout # this will not work, see the doc of np.add.at
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    pass
    H = Wh.shape[0]

    a = x.dot(Wx) + prev_h.dot(Wh) + b

    z_i = sigmoid(a[:,:H])
    z_f = sigmoid(a[:,H:2*H])
    z_o = sigmoid(a[:,2*H:3*H])
    z_g = np.tanh(a[:,3*H:])

    next_c = z_f * prev_c + z_i * z_g
    z_t = np.tanh(next_c)
    next_h = z_o * z_t

    cache = (z_i, z_f, z_o, z_g, z_t, prev_c, prev_h, Wx, Wh, x)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    pass
    H = dnext_h.shape[1]
    z_i, z_f, z_o, z_g, z_t, prev_c, prev_h, Wx, Wh, x = cache
  
    dz_o = z_t * dnext_h
    dc_t = z_o * (1 - z_t * z_t) * dnext_h + dnext_c
    dz_f = prev_c * dc_t
    dz_i = z_g * dc_t
    dprev_c = z_f * dc_t
    dz_g = z_i * dc_t
    
    da_i = (1 - z_i) * z_i * dz_i
    da_f = (1 - z_f) * z_f * dz_f
    da_o = (1 - z_o) * z_o * dz_o
    da_g = (1 - z_g * z_g) * dz_g
    da = np.hstack((da_i, da_f, da_o, da_g))
  
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    
    db = np.sum(da, axis = 0)
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    
    在一整个数据序列上面的LSTM的前向传播。我们假设一个由T个向量组成的序列，每个维度为D。这个LSTM
    使用一个大小为H的隐藏层，且我们在一个包含N序列的批量数据上面工作。在运行完LSTM的前向传播以后，
    我们返回所有的时间序列的隐藏层状态。

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    
    注意初始的cell状态是从输入传入的，但是初始的cell状态被设置为0了。也要注意cell状态不被返回的；
    这是LSTM的一个内部的状态且不会被从外部访问。

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    pass
    N, T, D = x.shape
    H = b.shape[0]//4
    h = np.zeros((N, T, H))
    cache = {}
    prev_h = h0
    prev_c = np.zeros((N, H))

    for t in range(T):
        xt = x[:, t, :]
        next_h, next_c, cache[t] = lstm_step_forward(xt, prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c       
        h[:, t, :] = prev_h
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    pass
    N, T, H = dh.shape
    z_i, z_f, z_o, z_g, z_t, prev_c, prev_h, Wx, Wh, x = cache[T-1]
    D = x.shape[1]
    
    dprev_h = np.zeros((N, H))
    dprev_c = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx= np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))
    
    for t in range(T):
        t = T-1-t
        step_cache = cache[t]
        dnext_h = dh[:,t,:] + dprev_h
        dnext_c = dprev_c
        dx[:,t,:], dprev_h, dprev_c, dWxt, dWht, dbt = lstm_step_backward(dnext_h, dnext_c, step_cache)
        dWx, dWh, db = dWx+dWxt, dWh+dWht, db+dbt
    
    dh0 = dprev_h  
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    
    时间放射层的前向传播。这个输入是一组D维N个时间序列的向量，每个长度为T。我们使用一个仿射函数
    来讲这些向量变为一个新的维度为M的向量

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
