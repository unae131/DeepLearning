"""
@ Binary classification using wide 2-layered net (cross-entropy loss)
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

    return [np.array(X, dtype=np.float128), np.array(Y, dtype=int)]

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

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def d_sigmoid(A, Y):
    return A - Y

def ReLU(Z):
    A = []
    for row in Z:
        a = []
        for z in row:
            if z <= 0:
                a.append(0)
            else:
                a.append(z)
        A.append(a)
    
    return np.array(A, dtype= np.float128)

def d_ReLU(A):
    dZ = []
    for row in A:
        dz = []
        for a in row:
            if a <= 0:
                dz.append(0)
            else:
                dz.append(1)
        dZ.append(dz)
    return np.array(dZ, dtype= np.float128)

def cross_entropy_loss(Y, Y_hat):
    m = len(Y)
    J = -(np.dot(Y, np.log10(Y_hat.T+EPSILON)) + np.dot((1 - Y), np.log10(1-Y_hat.T+EPSILON))) / m
    return J

def update(X, Y, K, alpha, W1, b1, W2, b2):
    m = len(X[0])
    
    for k in range(K):
        # forward
        Z1 = np.dot(W1.T, X) + b1
        A1 = ReLU(Z1)
        Z2 = np.dot(W2.T, A1) + b2
        A2 = sigmoid(Z2)

        J = cross_entropy_loss(Y, A2)

        # backward
        dZ2 = d_sigmoid(A2, Y)
        dW2 = np.dot(A1, dZ2.T) / m
        db2 = np.sum(dZ2) / m
        dZ1 = d_ReLU(A1)
        dW1 = np.dot(X, dZ1.T) / m
        db1 = np.sum(dZ1) / m
        
        # update
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        if k % 50 == 0 and k != K-1:
            print("W1: " + str(W1)) 
            print("b1: " + str(b1))
            print("W2: " + str(W2)) 
            print("b2: " + str(b2))

    print("W1: " + str(W1)) 
    print("b1: " + str(b1))
    print("W2: " + str(W2)) 
    print("b2: " + str(b2))
    print("J: " + str(J))

    return W1, b1, W2, b2, J

def test(X, Y, W1, b1, W2, b2):
    n = len(Y)

    Z1 = np.dot(W1.T, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2.T, A1) + b2
    A2 = sigmoid(Z2)

    J = cross_entropy_loss(Y, A2)
    
    # print("test J: " + str(J))

    return J

def predict(W1, b1, W2, b2, x):
    z1 = np.dot(W1.T, x) + b1
    a1 = [z if z > 0 else 0 for z in z1]
    z2 = np.dot(W2.T, a1) + b2
    a2 = sigmoid(z2)
    return int(a2 + 0.5)

def getAccuracy(W1, b1, W2, b2, X, Y):
    m = len(X[0])

    correct = 0
    for i in range(m):
        if predict(W1, b1, W2, b2, X[:,i]) == Y[i]:
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

    # Step 1. Load samples:
    X_train, Y_train = loadSamples(m, "train_samples.txt")
    X_test, Y_test = loadSamples(n, "test_samples.txt")

    W1 = np.array([[1619.75772464, 1420.44288296, 1423.28102635],
                    [2171.48707606, 1413.96673169, 1466.16849836]], dtype = np.float128)
    b1 = -76.141087000000002634
    W2 = np.array([-3.71621728, 63.48923154, 64.03892343], dtype = np.float128)
    b2 = -296.99185122385645214

    # vectorized
    # Step 2. Update W = [w1 , w2 ], b with 1000 samples for 2000 (=K) iterations: #K updates with the grad descent
    # Step 2-1. print W, b every 10 iterations
    start = time.time()
    W1, b1, W2, b2, cost = update(X_train, Y_train, K, alpha, W1, b1, W2, b2)

    # Step 2-2. calculate the cost on the 'm' train samples!
    print("cost for train samples : " + str(cost))

    # Step 2-3. calculate the cost with the 'n' test samples!
    cost = test(X_test, Y_test, W1, b1, W2, b2)
    print("cost for test samples : " + str(cost))

    # Step 2-4. print accuracy for the 'm' train samples! (display the number of correctly predicted outputs/m*100)
    print("accuracy for 'm' train samples: " + str(getAccuracy(W1, b1, W2, b2, X_train, Y_train)))
    # Step 2-5. print accuracy with the 'n' test samples! (display the number of correctly predicted outputs/n*100)
    print("accuracy for 'n' test samples: " + str(getAccuracy(W1, b1, W2, b2, X_test, Y_test)))
    
    print("time :", time.time() - start)


if __name__ == "__main__":
    main()