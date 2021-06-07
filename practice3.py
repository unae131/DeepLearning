"""
@ binary classification using wide 2-layered net implemented with tensorflow
"""
import tensorflow as tf
import numpy as np
import time
import os.path

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

"""
Input: 2-dim vector, 𝒙 = {𝑥1, 𝑥2}
Output: label of the input, y ∈ {0,1}
"""
m = 10000 # the number of train sample
n = 500 # the number of test sample
K = 5000 # the number of update

X_train, Y_train = loadSamples(m, "train_samples.txt")
X_test, Y_test = loadSamples(n, "test_samples.txt")

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, input_shape = [2,], activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd', loss = 'binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='sgd', loss = 'mse', metrics=['accuracy'])
# model.compile(optimizer='RMSprop', loss = 'binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

start = time.time()
# model.fit(X_train.T, Y_train, epochs=K, verbose = 0)
# model.fit(X_train.T, Y_train, epochs=K, verbose = 0, batch_size = 4)
# model.fit(X_train.T, Y_train, epochs=K, verbose = 0, batch_size = 32)
model.fit(X_train.T, Y_train, epochs=K, verbose = 0, batch_size = 128)

print("train time :", time.time() - start)

model.evaluate(X_train.T,  Y_train, verbose=2)
model.evaluate(X_test.T,  Y_test, verbose=2)