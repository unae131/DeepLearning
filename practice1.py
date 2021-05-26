"""
@ Compare the computation time with or without vectorization
"""

import numpy as np
import random
import math
import time

def generateSamples(m):
    x1, x2, y = [], [], []

    for i in range(m):
        x1.append(random.uniform(-10, 10))
        x2.append(random.uniform(-10, 10))
        
        if x1[-1] + x2[-1] > 0:
            y.append(1)
        else:
            y.append(0)

    return x1, x2, y

def updateSlowly(x1_train, x2_train, y_train, K, alpha, w, b):
    J = 0
    dw1 = 0
    dw2 = 0
    db = 0

    m = len(x1_train)
    
    z = np.zeros(m)
    a = np.zeros(m)
    dz = np.zeros(m)
    w1 = w[0]
    w2 = w[1]

    for k in range(K):
        for i in range(m):
            z[i] = w1*x1_train[i] + w2*x2_train[i] + b
            a[i] = 1/(1+np.exp(-z[i]))
            J += -(y_train[i]*math.log10(a[i]) + (1 - y_train[i])*math.log10(1-a[i]))
            dz[i] = a[i] - y_train[i]
            dw1 += x1_train[i]*dz[i]
            dw2 += x2_train[i]*dz[i]
            db += dz[i]

        J /= m
        dw1 /= m
        dw2 /= m
        db /= m

        # update
        w1 -= alpha * dw1
        w2 -= alpha * dw2
        b -= alpha * db

        if k % 10 == 0 and k != K-1:
            print("W: " + str([w1,w2])) 
            print("b: " + str(b))

    print("W: " + str([w1,w2]))
    print("b: " + str(b))
    print("J: " + str(J))

    w = [w1,w2]

    return w, b, J

def updateFastly(x1_train, x2_train, y_train, K, alpha, W, b):
    m = len(x1_train)
    
    X = np.array([x1_train, x2_train])
    Y = np.array(y_train).reshape(1, m)
    
    for k in range(K):
        Z = np.dot(W.T, X) + b
        A = 1 / (1 + np.exp(-Z))
        J = -(np.dot(Y, np.log10(A.T)) + np.dot((1 - Y), np.log10(1-A.T))) / m
        dZ = A - Y
        dW = np.dot(X, dZ.T) / m
        db = np.sum(dZ) / m
        
        # update
        W = W - alpha * dW
        b = b - alpha * db

        if k % 10 == 0 and k != K-1:
            print("W: " + str(W)) 
            print("b: " + str(b))

    print("W: " + str(W))
    print("b: " + str(b))
    print("J: " + str(J))

    return W, b, J

def slowTest(x1_test, x2_test, y_test, w, b):
    J = 0

    n = len(x1_test)
    
    z = np.zeros(n)
    a = np.zeros(n)

    w1 = w[0]
    w2 = w[1]

    for i in range(n):
        z[i] = w1*x1_test[i] + w2*x2_test[i] + b
        a[i] = 1/(1+np.exp(-z[i]))
        J += -(y_test[i]*math.log10(a[i]) + (1 - y_test[i])*math.log10(1-a[i]))

    J /= n

    # print("test J: " + str(J))

    return J

def fastTest(x1_test, x2_test, y_test, W, b):
    X = [x1_test, x2_test]
    Z = np.dot(W.T, X) + b
    a = 1 / (1 + np.exp(-Z))
    J = -(np.dot(y_test, np.log10(a).T) + np.dot((1 - np.array(y_test)), np.log10(1-a).T))
    
    J /= len(x1_test)

    # print("test J: " + str(J))

    return J

def predictSlowly(w, b, x):
    z = w[0]*x[0] + w[1]*x[1] + b
    y_hat = 1/(1+np.exp(-z))
    return int(y_hat + 0.5)

def predictFastly(W, b, x):
    z = np.dot(W.T, x) + b
    y_hat = 1/(1+np.exp(-z))
    return int(y_hat + 0.5)

def getAccuracySlowly(w, b, x1_train, x2_train, y_train):
    correct = 0
    for i in range(len(x1_train)):
        if predictSlowly(w, b, [x1_train[i], x2_train[i]]) == y_train[i]:
            correct += 1

    return correct / len(x1_train) * 100

def getAccuracyFastly(w, b, x1_train, x2_train, y_train):
    correct = 0
    for i in range(len(x1_train)):
        if predictFastly(w, b, [x1_train[i], x2_train[i]]) == y_train[i]:
            correct += 1

    return correct / len(x1_train) * 100
    
def main():
    """
    Input: 2-dim vector, 𝒙 = {𝑥1, 𝑥2}
    Output: label of the input, y ∈ {0,1}
    """
    m = 1000 # the number of train sample
    n = 100 # the number of test sample
    K = 2000 # the number of update
    alpha = 0.001

    # Step 1. Generate samples:
    x1_train, x2_train, y_train = generateSamples(m)
    x1_test, x2_test, y_test = generateSamples(n)

    W = np.array([0.000000000000000001,0.000000000000000001]).reshape(2,1)
    b = 0

    vectorized = True
    # vectorized = False

    if not vectorized :
        # Step 2. Update W = [w1 , w2 ], b with 1000 samples for 2000 (=K) iterations: #K updates with the grad descent
        # Step 2-1. print W, b every 10 iterations
        start = time.time()
        W, b, cost = updateSlowly(x1_train, x2_train, y_train, K, alpha, W, b)
        
        # Step 2-2. calculate the cost on the 'm' train samples!
        # cost = slowTest(x1_train, x2_train, y_train, w, b)
        print("cost for train samples : " + str(cost))

        # Step 2-3. calculate the cost with the 'n' test samples!
        cost = slowTest(x1_test, x2_test, y_test, W, b)
        print("cost for test samples : " + str(cost))

        # Step 2-4. print accuracy for the 'm' train samples! (display the number of correctly predicted outputs/m*100)
        print("accuracy for 'm' train samples: " + str(getAccuracySlowly(W, b, x1_train, x2_train, y_train)))
        # Step 2-5. print accuracy with the 'n' test samples! (display the number of correctly predicted outputs/n*100)
        print("accuracy for 'n' test samples: " + str(getAccuracySlowly(W, b, x1_test, x2_test, y_test)))
        
        print("element-wise time :", time.time() - start)

    else:
        # vectorized
        start = time.time()
        W, b, cost = updateFastly(x1_train, x2_train, y_train, K, alpha, W, b)

        # cost = fastTest(x1_train, x2_train, y_train, w, b)
        print("cost for train samples : " + str(cost))

        cost = fastTest(x1_test, x2_test, y_test, W, b)
        print("cost for test samples : " + str(cost))

        # Step 2-4. print accuracy for the 'm' train samples! (display the number of correctly predicted outputs/m*100)
        print("accuracy for 'm' train samples: " + str(getAccuracyFastly(W, b, x1_train, x2_train, y_train)))
        # Step 2-5. print accuracy with the 'n' test samples! (display the number of correctly predicted outputs/n*100)
        print("accuracy for 'n' test samples: " + str(getAccuracyFastly(W, b, x1_test, x2_test, y_test)))
        
        print("vectorized time :", time.time() - start)


if __name__ == "__main__":
    main()