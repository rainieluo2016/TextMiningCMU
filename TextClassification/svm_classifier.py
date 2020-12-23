import sys
import os
import datetime
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
import conjugateGradient as cg


# create mini batches
def get_mini_batches(X, y, batch_size):
    mini_batches = []
    # shuffle data
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    X = X[index]
    y = y[index]
    # number of batches
    n_batches = X.shape[0] // batch_size
    i = 0
    for i in range(n_batches + 1):
        X_batch = X[i * batch_size:(i + 1) * batch_size, :]
        Y_batch = y[i * batch_size:(i + 1) * batch_size, :]
        mini_batches.append((X_batch, Y_batch))
    # last batch
    if X.shape[0] % batch_size != 0:
        X_batch = X[i * batch_size:, :]
        Y_batch = y[i * batch_size:, :]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches


# compute loss (sparse matrix)
def svm_loss_sparse(X, y, w, lambd):
    # compute 1 - y w^T x (matrix form)
    margin = 1 - X.dot(w) * y
    # if < 0, won't be included in loss
    margin = np.where(margin < 0, 0, margin)
    loss = np.dot(w.transpose(), w) / 2 + lambd * margin.sum()
    # take mean
    loss /= X.shape[0]
    return loss


# compute gradient (sparse matrix)
def gradient_sparse(X, y, w, batch_size, lambd):
    # compute 1 - y w^T x (matrix form)
    margin = 1 - X.dot(w) * y
    # if < 0, won't be included in gradient
    mask = np.where(margin < 0, 0, 1)
    mask = csr_matrix(mask)
    # mean gradient
    grad = w - lambd * (mask.multiply(X.multiply(y)).sum(axis=0).reshape((-1, 1))) / batch_size
    # grad /= batch_size
    return grad


# mini batch gradient descent
def miniBatchGD(X, y, X_test, y_test, lambd, func_value, lr=0.001, beta=0.2, batch_size=1000, epochs=5):
    start = datetime.datetime.now()
    # initialization
    X = csr_matrix(X)
    w = np.zeros((X.shape[1], 1))
    func_diff_list = []
    test_accu_list = []
    grad_norm_list = []
    print('Start Training using Mini-Batch GD...')
    for epoch in range(epochs):
        # get mini batches
        batches = get_mini_batches(X, y, batch_size)
        # annealing learning rate update
        lr = lr / (1 + epoch * beta)
        # for each batch, update the weight
        for batch in batches:
            X_batch, y_batch = batch
            grad = gradient_sparse(X_batch, y_batch, w, batch_size, lambd)
            grad = np.array(grad).reshape(-1, 1)
            # compute gradient norm
            grad_norm_list.append(np.linalg.norm(grad))
            w = w - lr * grad
        # compute loss on the entire dataset
        loss = svm_loss_sparse(X, y, w, lambd)
        # function difference
        func_diff = (loss.item() - func_value) / func_value
        func_diff_list.append(func_diff)
        # test accuracy
        test_accu = test(X_test, y_test, w)
        test_accu_list.append(test_accu)
        print('Epoch', epoch+1, 'Test accuracy:', test_accu, 'Train loss:', loss.item())
    end = datetime.datetime.now()
    print('Finished training. Time elapsed: {0} s'.format((end - start).total_seconds()))
    return w, func_diff_list, grad_norm_list, test_accu_list


# compute test accuracy
def test(X, y, w):
    # if projection > 0, classify as positive, otherwise negative
    projection = np.dot(X, w)
    pred = np.where(projection >= 0, 1, -1)
    assert pred.shape[0] == y.shape[0]
    correct = np.count_nonzero(pred == y)
    return correct / len(pred)


# get I defined in the writeup for Newton's method
def get_I(X, y, w):
    margin = 1 - X.dot(w) * y
    I = np.nonzero(margin > 0)[0].reshape((-1, ))
    return I


# Newton's method
def newtonUpdate(X, y, X_test, y_test, lambd, func_value):
    start = datetime.datetime.now()
    # initialization
    X = csr_matrix(X)
    w = np.zeros((X.shape[1], 1))
    func_diff_list = []
    test_accu_list = []
    grad_norm_list = []
    epochs = 5
    print('Start Training using Newton\'s Method...')
    for epoch in range(epochs):
        I = get_I(X, y, w)
        grad = gradient_sparse(X, y, w, batch_size=X.shape[0], lambd=lambd)
        grad = np.array(grad)
        grad = grad.reshape(1, -1).flatten()
        # compute gradient norm
        grad_norm_list.append(np.linalg.norm(grad))
        # get update direction from conjugateGradient
        d, iter = cg.conjugateGradient(X, I, grad, lambd)
        d = d.reshape(-1, 1)
        w += d
        # test accuracy
        test_accu = test(X_test, y_test, w)
        test_accu_list.append(test_accu)
        # compute loss on the entire dataset
        loss = svm_loss_sparse(X, y, w, lambd)
        # function difference
        func_diff = (loss.item() - func_value) / func_value
        func_diff_list.append(func_diff)
        print('Epoch', epoch+1, 'Test accuracy:', test_accu, 'Train loss:', loss.item())
    end = datetime.datetime.now()
    print('Finished training. Time elapsed: {0} s'.format((end - start).total_seconds()))
    return w, func_diff_list, grad_norm_list, test_accu_list


# read and process svmlight data
def get_svmlight_data(train_path, test_path):
    # read raw data (X being sparse matrix)
    X_train, y_train = load_svmlight_file(train_path)
    X_test, y_test = load_svmlight_file(test_path)
    # make training data to dense and add the bias term
    X_train = csr_matrix.todense(X_train)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = csr_matrix.todense(X_test)
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    # reshape y to help with the computation
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    return X_train, y_train, X_test, y_test


def main():
    # read the train file from first arugment
    train_file = sys.argv[1]

    # read the test file from second argument
    test_file = sys.argv[2]
    # You can use load_svmlight_file to load data from train_file and test_file
    X_train, y_train, X_test, y_test = get_svmlight_data(train_file, test_file)
    # parse file name to determine the parameter values
    if 'covtype' in train_file and 'covtype' in test_file:
        lambd = 3631.3203125
        func_value = 2541.664519
    elif 'realsim' in train_file and 'realsim' in test_file:
        lambd = 7230.875
        func_value = 669.664812
    else:
        lambd = 0
        func_value = 0
        print('Neither covtype nor realsim data used!')
        return
    miniBatchGD(X_train, y_train, X_test, y_test, lambd=lambd, func_value=func_value)
    newtonUpdate(X_train, y_train, X_test, y_test, lambd=lambd, func_value=func_value)


# Main entry point to the program
if __name__ == '__main__':
    main()
