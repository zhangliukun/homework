from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
    """
    一个CaptioningRNN使用一个循环神经网络从图片特征产生标注

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.
    
    这个RNN接收大小为D的输入向量，有一个大小为V的词汇表，工作在一个长度为T的序列中。使用单词向量
    的维度为W，在大小为N的小批量数据上操作。

    注意我们不使用任何的正则化。
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.维度为D的输入图像的特征向量
        - wordvec_dim: Dimension W of word vectors.维度为W的单词向量
        - hidden_dim: Dimension H for the hidden state of the RNN.维度为H的RNN隐藏层
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # 初始化单词向量
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        # 初始化CNN->隐藏层映射参数
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # 初始化RNN的参数
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        # 初始化词汇权重的输出
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)


    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.
        
        计算RNN的训练时损失。我们为这些图片输入图片特征和真实标注，然后使用一个RNN来计算所有参数的损失
        和梯度。

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        
        # 将标注切分成两块：captions_in 除了最后一个单词其它都有且将会被输入进RNN；captions_out
        # 除了第一个单词其它都有且这是我们期望RNN生成的。这些会相互抵消因为RNN在接受word（t）以后
        # 应该产生word（t+1）。captions_in的第一个元素将会是START标志，且captions_out的第一个元素
        # 将会是第一个单词。
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        # 初始化隐藏层的图像特征的仿射变换的权重和偏置
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # Word embedding matrix
        W_embed = self.params['W_embed']

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        
        # 实现CaptioningRNN的前向和反向传播。在前向传播中你将会需要做以下：
        # 1. 使用仿射变换来从图片的特征计算初始的隐藏层状态。它会产生一个（N，H）的数组。      
        # 2. 使用一个word embedding层来将captions_in里面的单词从所以变为向量，为（N,T,W）的数组
        # 3. 使用vanilla RNN或者LSTM来为所有的时间步长执行输入单词向量的序列且产出隐藏层状态向量，为（N,T,H）的数组
        # 4. 使用一个（时间的）仿射变换来使用隐藏状态来在每一个时间步长上对词汇表计算得分，为（N,T,V）的数组
        # 5. 使用（时间的）softmax来使用captions_out计算损失，使用上面的mask来忽略输出单词是NULL的点
        
        
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        
        # 在反向传播中你将会需要计算所有模型参数的梯度的损失。使用上面定义的loss和grads变量来存储
        # 损失和梯度；grads[k]应该给出self.params[k]的梯度。
        ############################################################################
        pass
        #(1)
        affine_out, affine_cache = affine_forward(features ,W_proj, b_proj)
        #(2)
        word_embedding_out, word_embedding_cache = word_embedding_forward(captions_in, W_embed)
        #(3)
        if self.cell_type == 'rnn':
            rnn_or_lstm_out, rnn_cache = rnn_forward(word_embedding_out, affine_out, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            rnn_or_lstm_out, lstm_cache = lstm_forward(word_embedding_out, affine_out, Wx, Wh, b)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)
        #(4)
        temporal_affine_out, temporal_affine_cache = temporal_affine_forward(rnn_or_lstm_out, W_vocab, b_vocab)
        #(5)
        loss, dtemporal_affine_out = temporal_softmax_loss(temporal_affine_out, captions_out, mask)
        #(4)
        drnn_or_lstm_out, grads['W_vocab'], grads['b_vocab']=temporal_affine_backward(dtemporal_affine_out,temporal_affine_cache)
        #(3)
        if self.cell_type == 'rnn':
            dword_embedding_out, daffine_out, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(drnn_or_lstm_out, rnn_cache)
        elif self.cell_type == 'lstm':
            dword_embedding_out, daffine_out, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(drnn_or_lstm_out, lstm_cache)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)
        #(2)
        grads['W_embed'] = word_embedding_backward(dword_embedding_out, word_embedding_cache)
        #(1)
        dfeatures, grads['W_proj'], grads['b_proj'] = affine_backward(daffine_out, affine_cache)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.
        
        为模型运行一个测试时前向传播，为输入特征向量采样标注。

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.
        
        在每个时间步长中，我们嵌入当前的单词，传递它和先前的隐藏状态给RNN来得到下一个隐藏状态，使用
        隐藏状态来得到所有单词的得分，然后选择最高的分数的单词作为下一个单词，初始隐藏状态是通过将
        一个仿射变换应用到输入图片特征上面计算出来的，然后这个初始单词是START标记。

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        ###########################################################################
        pass
        N, D = features.shape
        affine_out, affine_cache = affine_forward(features ,W_proj, b_proj)
    
        prev_word_idx = [self._start]*N
        prev_h = affine_out
        prev_c = np.zeros(prev_h.shape)
        captions[:,0] = self._start
        for i in range(1,max_length):
            prev_word_embed  = W_embed[prev_word_idx]
            if self.cell_type == 'rnn':
                next_h, rnn_step_cache = rnn_step_forward(prev_word_embed, prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                next_h, next_c,lstm_step_cache = lstm_step_forward(prev_word_embed, prev_h, prev_c, Wx, Wh, b)
                prev_c = next_c
            else:
                raise ValueError('Invalid cell_type "%s"' % self.cell_type)
            vocab_affine_out, vocab_affine_out_cache = affine_forward(next_h, W_vocab, b_vocab)
            captions[:,i] = list(np.argmax(vocab_affine_out, axis = 1))
            prev_word_idx = captions[:,i]
            prev_h = next_h
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
