"""
@ Binary classification using logistic regression(cross-entropy loss)
"""

import numpy as np
import random
import math
import time
import os.path
EPSILON = 0.0000000000000000001

def loadSamples(m, fileName):
    if not os.path.exists(fileName):
        return generateSamples(m, fileName)

    with open(fileName, 'r') as f:
        lines = f.readlines()

        X = []
        for line in lines[:-1]:
            X.append(line.split())

        Y = lines[-1].split()

    return [np.array(X, dtype=float), np.array(Y, dtype=int)]

def saveSamples(X, Y, fileName):
    lines = ""

    for i in range(len(X)):
        for j in range(len(X[0])):
            lines += str(X[i][j]) + " "
        
        lines = lines[:-1] + "\n"

    for i in range(len(Y)):
        lines += str(Y[i]) + " "
    lines = lines[:-1]

    with open(fileName, 'w') as f:
        f.write(lines)

def generateSamples(m, fileName):
    x1, x2, y = [], [], []

    for i in range(m):
        x1.append(random.uniform(-10, 10))
        x2.append(random.uniform(-10, 10))

        if x1[-1] + x2[-1] > 0:
            y.append(1)
        else:
            y.append(0)

    X = np.array([x1, x2])
    Y = np.array(y)

    saveSamples(X, Y, fileName)

    return X, Y

def cross_entropy_loss(Y, Y_hat):
    m = len(Y)
    J = -(np.dot(Y, np.log10(Y_hat.T+EPSILON)) + np.dot((1 - Y), np.log10(1-Y_hat.T+EPSILON))) / m
    return J

def update(X, Y, K, alpha, W, b):
    m = len(X[0])
    
    for k in range(K):
        Z = np.dot(W.T, X) + b
        A = 1 / (1 + np.exp(-Z))
        J = cross_entropy_loss(Y, A)
        dZ = A - Y
        dW = np.dot(X, dZ.T) / m
        db = np.sum(dZ) / m
        
        # update
        W = W - alpha * dW
        b = b - alpha * db

        if k % 50 == 0 and k != K-1:
            print("W: " + str(W)) 
            print("b: " + str(b))

    print("W: " + str(W))
    print("b: " + str(b))
    print("J: " + str(J))

    return W, b, J

def test(X, Y, W, b):
    m = len(X[0])

    Z = np.dot(W.T, X) + b
    A = 1 / (1 + np.exp(-Z))
    J = cross_entropy_loss(Y, A)

    # print("test J: " + str(J))

    return J

def predict(W, b, x):
    z = np.dot(W.T, x) + b
    a = 1/(1+np.exp(-z))
    return int(a + 0.5)

def getAccuracy(W, b, X, Y):
    m = len(X[0])

    correct = 0
    for i in range(m):
        if predict(W, b, X[:,i]) == Y[i]:
            correct += 1

    return correct / m * 100
    
def main():
    """
    Input: 2-dim vector, 𝒙 = {𝑥1, 𝑥2}
    Output: label of the input, y ∈ {0,1}
    """
    m = 10000 # the number of train sample
    n = 500 # the number of test sample
    K = 5000 # the number of update
    alpha = 0.01

    # Step 1. Generate samples:
    X_train, Y_train = loadSamples(m, "train_samples.txt")
    X_test, Y_test = loadSamples(n, "test_samples.txt")

    W = np.array([0.000000000000000001,0.000000000000000001]).reshape(2,1)
    b = 0

    # vectorized
    # Step 2. Update W = [w1 , w2 ], b with 1000 samples for 2000 (=K) iterations: #K updates with the grad descent
    # Step 2-1. print W, b every 10 iterations
    start = time.time()
    W, b, cost = update(X_train, Y_train, K, alpha, W, b)

    # Step 2-2. calculate the cost on the 'm' train samples!
    # cost = fastTest(x1_train, x2_train, y_train, w, b)
    print("train time :", time.time() - start)
    print("cost for train samples : " + str(cost))

    # Step 2-3. calculate the cost with the 'n' test samples!
    start = time.time()
    cost = test(X_test, Y_test, W, b)
    print("test time :", '{0:.6f}'.format(time.time() - start))
    print("cost for test samples : " + str(cost))

    # Step 2-4. print accuracy for the 'm' train samples! (display the number of correctly predicted outputs/m*100)
    print("accuracy for 'm' train samples: " + str(getAccuracy(W, b, X_train, Y_train)))
    # Step 2-5. print accuracy with the 'n' test samples! (display the number of correctly predicted outputs/n*100)
    print("accuracy for 'n' test samples: " + str(getAccuracy(W, b, X_test, Y_test)))
    


if __name__ == "__main__":
    main()