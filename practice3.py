"""
@ binary classification using wide 2-layered net implemented with tensorflow
"""
import tensorflow as tf
import numpy as np
import time

def loadSamples(m, fileName):
    try:
        with open(fileName, 'r') as f:
            lines = f.readlines()

            X = []
            for line in lines[:-1]:
                X.append(line.split()[:m])

            Y = lines[-1].split()

        return [np.array(X, dtype=np.float128), np.array(Y, dtype=int)]

    except FileNotFoundError:
        print(fileName, "No file")

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
# model.fit(X_train.T, Y_train, epochs=10, verbose = 0)
model.fit(X_train.T, Y_train, epochs=10, verbose = 0, batch_size = 4)
# model.fit(X_train.T, Y_train, epochs=10, verbose = 0, batch_size = 32)
# model.fit(X_train.T, Y_train, epochs=10, verbose = 0, batch_size = 128)

print("train time :", time.time() - start)

model.evaluate(X_train.T,  Y_train, verbose=2)
model.evaluate(X_test.T,  Y_test, verbose=2)