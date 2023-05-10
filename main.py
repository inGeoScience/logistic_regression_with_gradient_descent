import numpy
import pandas
from matplotlib import pyplot


# sigmoid function
def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


# cost function for logistic regression
def compute_cost(X, y, w_in, b_in):
    """
    compute the cost over all examples
    :param X: (ndarray Shape(m, n)) dataset, m samples by n features
    :param y: (array_like Shape(m,)) target values for all samples
    :param w_in: (array_like Shape(n,)) values of parameters of the model
    :param b_in: scalar Values of bias parameter of the model
    :return: the loss
    """
    m, n = X.shape
    cost = 0
    for i in range(m):
        cost += (y[i] * numpy.log(sigmoid(numpy.dot(w_in, X[i]) + b_in)) + (1 - y[i]) * numpy.log(1 - sigmoid(numpy.dot(w_in, X[i]) + b_in)))
    return (-1 / m) * cost


def compute_gradient(X, y, w_in, b_in):
    """
    compute gradient for each iteration
    :param X: (ndarray Shape(m, n)) dataset, m samples by n features
    :param y: (array_like Shape(m,)) target values for all samples
    :param w_in: (array_like Shape(n,)) values of parameters of the model
    :param b_in: scalar Values of bias parameter of the model
    :return: the gradient dj_dw and dj_db
    """
    m, n = X.shape
    dj_dw = numpy.zeros(n)
    dj_db = 0
    for j in range(n):
        for i in range(m):
            dj_dw[j] += (sigmoid(numpy.dot(w_in, X[i]) + b_in) - y[i]) * X[i][j]
    dj_dw /= m
    for i in range(m):
        dj_db += (sigmoid(numpy.dot(w_in, X[i]) + b_in) - y[i])
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, iters):
    """
    run gradient descent
    :param X: (ndarray Shape(m, n)) dataset, m samples by n features
    :param y: (array_like Shape(m,)) target values for all samples
    :param w_in: (array_like Shape(n,)) values of parameters of the model
    :param b_in: scalar value of bias parameter of the model
    :param alpha: learning rate α
    :param iters: iterations for gradient descent
    :return:
    """
    m_in, n_in = X.shape
    for inum in range(iters):
        dj_dw, dj_db = compute_gradient(X, y, w_in, b_in)
        for j in range(n_in):
            w_in[j] = w_in[j] - alpha * dj_dw[j]
        b_in = b_in - alpha * dj_db
    loss_in = compute_cost(X, y, w_in, b_in)
    return w_in, b_in, loss_in


def predict(X_pred, w_pred, b_pred):
    """
    make predictions with learned w and b
    :param X_pred: data set with m samples and n features
    :param w_pred: values of parameters of the model
    :param b_pred: scalar value of bias parameter of the model
    :return:
    """
    predictions = sigmoid(numpy.dot(X_pred, w_pred) + b_pred)
    p = [1 if item >= 0.5 else 0 for item in predictions]
    return numpy.array(p)


# 导入数据到一个DataFrame
data = pandas.read_csv('data/data.txt', header=None, names=['x1', 'x2', 'target'])
# 获取列数（包括输出变量）
colNum = data.shape[1]
# 定义数据集（不含输出变量）
X_train = data.iloc[:, :colNum - 1]
X_train = numpy.array(X_train.values)
# 定义输出空间
y_train = data.iloc[:, colNum - 1: colNum]
y_train = numpy.array(y_train.values)
# 获取样本量和特征数
m, n = X_train.shape
numpy.random.seed(1)
w_init = 0.01 * (numpy.random.rand(2).reshape(-1) - 0.5)
b_init = -8
w, b, loss = gradient_descent(X_train, y_train, w_init, b_init, alpha=0.001, iters=1)
print(f'w = {w}\nb = {b}\nloss = {loss}')
pred = predict(X_train, w, b)
print('Train Accuracy: %f'%(numpy.mean(pred == y_train.reshape(-1)) * 100))
