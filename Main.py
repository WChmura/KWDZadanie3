from keras.datasets import mnist
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()



print("Datasets size")
print("Train data:", X_train.shape)
print("Test data:", X_test.shape)

print("Samples from training data:")
for i in range(0,10):
    plt.subplot(1,10,i+1)
    plt.imshow(X_train[i], cmap=plt.get_cmap("gray"))
    plt.title(y_train[i]);
    plt.axis('off');