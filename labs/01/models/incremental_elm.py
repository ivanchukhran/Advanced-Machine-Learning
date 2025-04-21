import numpy as np
from sklearn.preprocessing import StandardScaler

class IncrementalELM:
    """
    Implementation of Incremental Extreme Learning Machine (IELM)
    """
    def __init__(self, n_hidden=50, activation='sigmoid'):
        self.n_hidden = n_hidden
        self.activation = activation
        self.initialized = False
        self.W = None  # Input to hidden weights
        self.beta = None  # Hidden to output weights
        self.scaler = StandardScaler()

    def _initialize(self, X):
        n_features = 1  # For the case of our 1D data
        # Randomly initialize input weights and biases
        self.W = np.random.randn(n_features, self.n_hidden)
        self.bias = np.random.randn(self.n_hidden)
        self.beta = np.zeros((self.n_hidden, 1))
        self.initialized = True

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _activation(self, x):
        if self.activation == 'sigmoid':
            return self._sigmoid(x)
        # Can add more activation functions if needed
        return x

    def _hidden_output(self, X):
        # Calculate the output of the hidden layer
        G = self._activation(np.dot(X, self.W) + self.bias)
        return G

    def learn_one(self, x: dict[str, float], y: float):
        # Convert inputs to proper format

        X = np.array([[x['x']]])
        y_val = 1 if y == 1 else 0  # Convert to binary for this example
        y = np.array([[y_val]])

        if not self.initialized:
            self._initialize(X)

        # Calculate hidden layer output
        H = self._hidden_output(X)

        # Update output weights using sequential learning formula
        # This is a simplified version of the Online Sequential ELM (OS-ELM) algorithm
        learning_rate = 0.1
        y_pred = np.dot(H, self.beta)
        error = y - y_pred
        delta = learning_rate * H.T * error
        self.beta += delta

        return self

    def predict_one(self, x):
        if not self.initialized:
            return 0  # Return default value if not initialized

        X = np.array([[x['x']]])
        H = self._hidden_output(X)
        y_pred = np.dot(H, self.beta)

        # For binary classification
        return 1 if y_pred >= 0.5 else 0
