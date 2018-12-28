from keras.datasets import mnist
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

samples, x, y = X_train.shape
X_train = X_train.reshape((samples,x*y))

samples, x, y = X_test.shape
X_test = X_test.reshape((samples,x*y))

neural_network = MLPRegressor(hidden_layer_sizes=(30,20,10,), activation='logistic', solver="sgd",
                              batch_size=20, shuffle=True, momentum=0.95, alpha=1e-5, verbose=False,
                              max_iter=500, tol=1e-7, random_state=1)

neural_network.fit(X_train, y_train)

plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.imshow(np.transpose(neural_network.coefs_[0]), cmap=plt.get_cmap("gray"), aspect="auto")
plt.ylabel('neurons in first hidden layer');
plt.xlabel('input weights to neural network');

print("Mean squared error of a learned neural network model: %.2f" %
      mean_squared_error(y_test, neural_network.predict(X_test)))


print('Variance score for neural network model: %.2f' % r2_score(y_test, neural_network.predict(X_test)))

