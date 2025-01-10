import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, activation_function, validation_split):
        self.L = len(layers)
        self.n = layers.copy()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.fact = activation_function
        self.validation_split = validation_split

        # Initialize arrays for network components
        self.train_errors = []
        self.val_errors = []
        self.h = [np.zeros(n) for n in layers]
        self.xi = [np.zeros(n) for n in layers]
        self.w = [np.zeros((1, 1))] + [np.random.uniform(-1.0, 1.0, (layers[i], layers[i-1])) * np.sqrt(1 / layers[i-1]) for i in range(1, self.L)]
        self.theta = [np.zeros(n) for n in layers]
        self.delta = [np.zeros(n) for n in layers]
        self.d_w = [np.zeros_like(self.w[i]) for i in range(self.L)]
        self.d_theta = [np.zeros_like(self.theta[i]) for i in range(self.L)]
        self.d_w_prev = [np.zeros_like(self.w[i]) for i in range(self.L)]
        self.d_theta_prev = [np.zeros_like(self.theta[i]) for i in range(self.L)]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Split data into training and validation sets (80% training, 20% validation)
        split_index = int(n_samples * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        for epoch in range(self.epochs):
            for pat in range(len(X_train)):
                # Choose a random pattern from the training set
                idx = np.random.randint(0, len(X_train))
                x_mu, z_mu = X_train[idx], y_train[idx]

                # Forward propagation
                self._forward_propagate(x_mu)

                # Backpropagation
                self._back_propagate(z_mu)

                # Update weights and thresholds
                self._update_weights()

            # Calculate training and validation error
            train_error = self._calculate_error(X_train, y_train)
            val_error = self._calculate_error(X_val, y_val)
            self.train_errors.append(train_error)
            self.val_errors.append(val_error)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Error: {train_error}, Validation Error: {val_error}")

    def predict(self, X):
        n_samples, n_features = X.shape
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            self._forward_propagate(X[i])
            y_pred[i] = self.xi[-1]  # Output layer activation
        return y_pred

    def loss_epochs(self):
        training_error = np.zeros(self.epochs)
        validation_error = np.zeros(self.epochs)
        # Track and return training and validation error evolution per epoch
        return training_error, validation_error

    def _forward_propagate(self, x):
        # Set input layer activations
        self.xi[0] = np.array(x, dtype=float)  # Ensure x is a NumPy array
        # Propagate through layers
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l], self.xi[l-1]) - self.theta[l]  
            self.xi[l] = self._activation(self.h[l])

    def _back_propagate(self, z):
        # Compute output layer error
        self.delta[-1] = (self.xi[-1] - z) * self._activation_derivative(self.h[-1])
        # Propagate error backwards through hidden layers
        for l in range(self.L-2, 0, -1):
            self.delta[l] = np.dot(self.w[l+1].T, self.delta[l+1]) * self._activation_derivative(self.h[l])

    def _update_weights(self):
        # Update weights and thresholds using gradient descent with momentum
        for l in range(1, self.L):
            self.d_w[l] = -self.learning_rate * np.outer(self.delta[l], self.xi[l-1]) + self.momentum * self.d_w_prev[l]
            self.d_theta[l] = -self.learning_rate * self.delta[l] + self.momentum * self.d_theta_prev[l]
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]
            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l] = self.d_theta[l]

    def _activation(self, x):
        if self.fact == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip values to prevent overflow
        elif self.fact == 'relu':
            return np.maximum(0, x)
        elif self.fact == 'linear':
            return x
        elif self.fact == 'tanh':
            return np.tanh(np.clip(x, -500, 500))  # Clip values to prevent overflow
        else:
            raise ValueError("Unsupported activation function")

    def _activation_derivative(self, x):
        if self.fact == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return sig * (1 - sig)
        elif self.fact == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.fact == 'linear':
            return np.ones_like(x)
        elif self.fact == 'tanh':
            return 1 - np.tanh(np.clip(x, -500, 500)) ** 2
        else:
            raise ValueError("Unsupported activation function")

    def _calculate_error(self, X, y):
        total_error = 0
        for i in range(len(X)):
            self._forward_propagate(X[i])
            total_error += 0.5 * np.sum((self.xi[-1] - y[i]) ** 2)
        return total_error / len(X)
    
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mae, mape

# Load training data from CSV file
train_data = pd.read_csv('train_data.csv')

# Convert categorical features to dummy variables
train_data = pd.get_dummies(train_data)

# Separate features and target variable
X = train_data.drop(columns=['CO2 Emissions(g/km)']).values  # Assuming 'CO2 Emissions(g/km)' is the target variable
y = train_data['CO2 Emissions(g/km)'].values

# Example usage
layers = [X.shape[1], 9, 5, 1]  # Network architecture: input layer size based on data, 9 and 5 hidden, 1 output
learning_rate = 0.01
momentum = 0.9
activation_function = 'sigmoid'
epochs = 30

# Initialize and train the model
neural_net = NeuralNet(layers, epochs=epochs, learning_rate=learning_rate, momentum=momentum, activation_function=activation_function, validation_split=0.2)
neural_net.fit(X, y)

# Save evaluation metrics
train_error = neural_net._calculate_error(X, y)
mse, mae, mape = evaluate(y, neural_net.predict(X))
hyperparameter_results = []
hyperparameter_results.append({
    'Number of layers': len(layers),
    'Layer Structure': layers,
    'Num epochs': epochs,
    'Learning Rate': learning_rate,
    'Momentum': momentum,
    'Activation function': activation_function,
    'Training Error': train_error,
    'MSE': mse,
    'MAE': mae,
    'MAPE': mape
})

# Convert results to DataFrame
results_df = pd.DataFrame(hyperparameter_results)
print(results_df)

# Save the trained model
with open('neural_net_model.pkl', 'wb') as file:
    pickle.dump(neural_net, file)
print("Model saved successfully.")

print("L =", neural_net.L)
print("n =", neural_net.n)
print("Weights for layer 2:", neural_net.w[2])
