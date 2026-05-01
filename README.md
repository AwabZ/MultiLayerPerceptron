# 🧠 MNIST-MLP: Deep Learning from First Principles
This project features a ground-up implementation of a Multilayer Perceptron (MLP) designed to classify handwritten digits from the MNIST dataset. The codebase is heavily inspired by and built upon the foundational educational material in Michael Nielsen's book, "Neural Networks and Deep Learning".

# 📖 Inspiration
The architecture and mathematical implementation of this network—specifically the backpropagation logic and stochastic gradient descent, derive from the "first principles" approach championed by Michael Nielsen. By avoiding high-level libraries like TensorFlow or PyTorch, this project focuses on the raw linear algebra and calculus required to train a functional neural network.

# 🛠️ Technical Architecture
### 1. The Network Core (network.py):
- The Network class represents the brain of the project. It supports an arbitrary number of layers and utilizes the Sigmoid activation function for all neurons.  
- Weights & Biases: Initialized using a standard Gaussian distribution ($\mathcal{N}(0, 1)$).  
- Feedforward: Propagates an input vector $a$ through the network using the relation $a' = \sigma(wa + b)$.  
- Backpropagation: The core engine for calculating the gradient of the cost function. it computes the error $\delta$ for the output layer and back-propagates it through the hidden layers to find the partial derivatives for weights and biases.

### 2. Training Engine: Stochastic Gradient Descent (SGD)
The network is optimized using Stochastic Gradient Descent (SGD):  

- Shuffling: Training data is shuffled every epoch to prevent biased learning.  
- Mini-Batches: Updates are performed on small subsets of data to increase computational efficiency and convergence stability.  
- Hyperparameters: Users can tune the number of epochs, batch size, and the learning rate (eta).

### 3. Data Pipeline: (MLP_MNIST_Loader.py):
To handle the MNIST dataset, a specialized loader performs necessary data transformations:  
- Gzip/Pickle: Extracts the compressed binary data.
- Vectorization: Converts digit labels (e.g., 5) into a one-hot encoded vector (a 10-dimensional vector with 1.0 at index 5) for training.
- Reshaping: flattens the $28 \times 28$ pixel images into a 784-dimensional input vector. 
