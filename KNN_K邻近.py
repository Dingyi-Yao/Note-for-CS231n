import numpy as np

## 第一讲
def load_CIFAR10(path):
    """
    需要自己定义，返回值:
    训练集Xtr 50000 x 32 x 32 x 3 
    训练集标签Ytr 50,000 x 1
    测试集与测试集标签te
    """
    Xtr, Ytr, Xte, Yte =0, 0, 0, 0
    return Xtr, Ytr, Xte, Yte
class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """
        其中X是NxD，而y是一维列向量
        """
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ 
        X 是 N x D ，每一行是待测试 
        """
        num_test = X.shape[0]
    # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
        return Ypred

#调用
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # load函数具体形式需要自己未定义
# 将矩阵展成一维
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # 10000 x 3072
nn = NearestNeighbor() # 创建类的实例
nn.train(Xtr_rows, Ytr) # 训练
Yte_predict = nn.predict(Xte_rows) # 预测
print ('accuracy: %f'  % ( np.mean(Yte_predict == Yte) ) )