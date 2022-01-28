from audioop import bias
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, num_epochs):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def fit(self, features, targets):

        # Initializing all the weights to 0
        self.features = features
        self.targets = targets
        self.feature_len = features.shape[1]
        self.train_rows = features.shape[0]
        self.weights = np.zeros(self.feature_len)
        self.bias = 0

        for i in range(self.num_epochs):
            self.updateWeights()
        return self

    def predict(self, feature):
        return feature.dot(self.weights)+self.bias

    def updateWeights(self):
        predicted_values = self.predict(self.features)

        weight_gradient = - \
            (2*(self.features.T).dot(self.targets-predicted_values))/self.train_rows

        bias_gradient = -2 * \
            (np.sum(self.targets-predicted_values))/self.train_rows

        self.weights = self.weights - self.learning_rate*weight_gradient
        self.bias = self.bias - self.learning_rate*bias_gradient

        return self


# Train Data
features = np.array([1, 2, 3, 4, 5, 6, 7])
targets = np.array([2, 4, 6, 8, 10, 12, 14])

# Test Data
predict_features = np.array([1.5, 2.5, 3.5, 4.5, 100])
predicted_actual_outputs = np.array([3, 5, 7, 9, 200])

# Reshaping to desired shape
features = features.reshape((-1, 1))
predict_features = predict_features.reshape((-1, 1))

# LinearRegression Object
lr = LinearRegression(0.01, 11)
lr.fit(features, targets)
print("Weights: ", lr.weights)
print("Bias: ", lr.bias)


print("Predicted Outputs: ", lr.predict(predict_features))
print("Original Outputs:", predicted_actual_outputs)

# Plotting Logic
plt.plot(features, targets, label="Original Data")
plt.plot(features, lr.weights*features+lr.bias, label="Fitted Data")
plt.legend(loc="upper left")
plt.show()
