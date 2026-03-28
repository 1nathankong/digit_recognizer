import pandas as pd
from matplotlib import pyplot as plt
import cupy as cp

data = pd.read_csv('train.csv')

data = cp.array(data)

m , n = data.shape
cp.random.shuffle(data)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train.astype('float16') / 255
X_dev = X_dev.astype('float16') / 255

def init_params():
    W1 = cp.random.randn(512,784) * cp.sqrt(2.0/784) # the orignal - 0.5 is  just a educated guess values 
    b1 = cp.random.randn(512,1) 
    
    W2 = cp.random.randn(256,512) * cp.sqrt(2.0/512)
    b2 = cp.random.randn(256,1) 
    
    W3 = cp.random.randn(10,256) * cp.sqrt(2.0/256)
    b3 = cp.random.randn(10,1)
    
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return cp.maximum(0,Z)

def softmax(Z):
    exp_Z = cp.exp(Z - cp.max(Z, axis=0, keepdims=True)) 
    return exp_Z / cp.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = cp.zeros((Y.size, 10))
    one_hot_Y[cp.arange(Y.size), Y] = 1
    return one_hot_Y.T

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ3 = A3 - one_hot_Y
    dW3 = 1/m * dZ3.dot(A2.T)
    db3 = 1/m * cp.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * cp.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * cp.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3   

def get_predictions(A3):
    return cp.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return cp.sum(predictions == Y) / Y.size

def mini_batch_sgd(X,Y, epochs, alpha, batch_size = 64):
    W1, b1, W2, b2, W3, b3 = init_params()
    m = X.shape[1]

    for epoch in range(epochs):
        
        permutation = cp.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[permutation]
        alpha *= .9

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i : i + batch_size]
            Y_batch = Y_shuffled[i : i + batch_size]

            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_batch)
            dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X_batch, Y_batch)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        print(f"Epoch: {epoch}, Accuracy: {get_accuracy(get_predictions(A3), Y_batch)}, New Alpha: {alpha}")
    return W1, b1, W2, b2, W3, b3


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _,_,_,_,_,A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def test_predictions(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:,index, None]
    predictions = make_predictions(X_train[:,index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("prediction: ", predictions)
    print("label: ", label)

    current_image = current_image.get().reshape((28,28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


X_train_gpu = cp.asarray(X_train)
Y_train_gpu = cp.asarray(Y_train)
W1, b1, W2, b2, W3, b3 = mini_batch_sgd(X_train_gpu, Y_train_gpu, 6, .1, 50)



test_predictions(0, W1, b1, W2, b2, W3, b3)
test_predictions(1, W1, b1, W2, b2, W3, b3)

