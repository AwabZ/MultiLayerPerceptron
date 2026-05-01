import pickle
import gzip
import numpy as np

def load_mnist_data():
    with gzip.open('neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    train_data, val_data, test_data = load_mnist_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    training_results = [vectorized_results(y) for y in train_data[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    validation_data = list(zip(validation_inputs, val_data[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return (training_data, validation_data, test_data)

def vectorized_results(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e



