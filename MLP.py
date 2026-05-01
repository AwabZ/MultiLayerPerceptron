import random
import numpy as np
import MLP_MNIST_Loader

class Network:

    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.biases = []
        for y in layers[1:]:
            self.biases.append(np.random.randn(y, 1))
        self.weights = []
        for x, y in zip(layers[:-1], layers[1:]):
            self.weights.append(np.random.randn(y, x))

    
    def feedforward(self, a):
        """
        The Initial (a) is the input vector (x) and the final
        returned (a) is the output vector of the entire network
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a) + b)
        return a
    
    def SGD(self, training_data, epochs, batch_size, eta, test_data = None):
        if test_data:
            num_tests = len(test_data)
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in range(0, n, batch_size):
                mini_batches.append(training_data[k:k+batch_size])
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                test_results = self.evaluate(test_data)
                print(f"Epoch {epoch}: {test_results} / {num_tests}")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch, eta):
        n = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            single_nabla_b, single_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, single_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, single_nabla_w)]

        self.weights = [w - (eta/n) * nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/n) * nb for b,nb in zip(self.biases, nabla_b)]


    def evaluate(self, test_data):
        test_results = [ (np.argmax(self.feedforward(x)), y) for (x,y) in test_data ]
        return sum(int (x==y) for (x,y) in test_results)
    

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives: partial C_x / partial a 
        for the output activations
        """
        return (output_activations-y)
    



def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))



training_data, validation_data, test_data = MLP_MNIST_Loader.load_data_wrapper()
net = Network([784, 100, 50, 10])
net.SGD(training_data, 50, 10, 3.0, test_data=test_data)