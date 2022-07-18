import numpy as np

##第二讲

def L_i(x, y, W):
    """
    没有向量化，只是单纯计算一个样本的svm损失
    - x 图像矩阵，但是最后加了一个偏置，以简化形式，不再有b (e.g. 3073 x 1 in CIFAR-10)
    - y 整数，给出正确分类的索引
    - W 权重矩阵 (e.g. 10 x 3073 in CIFAR-10)
    """
    delta = 1.0 # 超参数
    scores = W.dot(x) # scores 10 x 1
    correct_class_score = scores[y]
    D = W.shape[0] # 类的种类, e.g. 10
    loss_i = 0.0
    for j in range(D): # 遍历所有错误的类
        if j == y:
        # 跳过分类正确的
            continue
        # 叠加损失
        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i

def L_i_vectorized(x, y, W):
    """
    半向量化，计算一个样本(x,y)，在函数中没有for循环，但是在函数外还需要一个
    """
    delta = 1.0 #超参数
    scores = W.dot(x)
    # compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)
    # on y-th position scores[y] - scores[y] canceled and gave delta. We want
    # to ignore the y-th position and only consider margin on max wrong class
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

def svm_loss_naive(W, X, y, reg):
    '''
    用两层循环计算全部样本
    reg: 正则化系数
    输出dW为梯度
    '''
    delta = 1.0 #超参数
    dW = np.zeros(W.shape) 
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
        margin = scores[j] - correct_class_score + delta # 
        #理论基础https://zhuanlan.zhihu.com/p/21478575?refer=baina
        if margin > 0:
            loss += margin
            dW[:, y[i]] += -X[i, :]     # 对于正确分类，梯度减去x
            dW[:, j] += X[i, :]         # 对于错误分类，梯度加上x
    loss /= num_train
    dW /= num_train
    dW += 2 * reg * W #除以N加上正则项
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    
    """
    全向量化 :
    这里定义和单个样本的不一样!!!
    - X 包括了所有样本，每一行是一个样本 (e.g. 50,000 x 3073 in CIFAR-10)
    - y 表示正确类的数组，标签 (e.g. 50,000-D array)
    - W 权重 (e.g. 3073 x 10)
    """
    delta = 1.0 #超参数
    loss = 0.0
    dW = np.zeros(W.shape) # 初始化
    scores = X.dot(W)  # N xC
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores_correct = scores[np.arange(num_train), y] #1 x N
    scores_correct = np.reshape(scores_correct, (num_train, 1)) # N x 1
    margins = scores - scores_correct + delta # N x C
    margins[np.arange(num_train), y] = 0.0 #分类正确的位置，不再加delta
    margins[margins <= 0] = 0.0 #用这个取代了max函数
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    margins[margins > 0] = 1.0                         # 对于错误分类，梯度减去x
    row_sum = np.sum(margins, axis=1)                  # 1 by N
    margins[np.arange(num_train), y] = -row_sum        #对于正确分类，梯度减去x
    dW += np.dot(X.T, margins)/num_train + reg * W     # D by C  # reg * W 是定义的问题
    return loss, dW