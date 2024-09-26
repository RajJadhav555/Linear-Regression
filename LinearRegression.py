import numpy as np


class LinearRegression:
    def __init__(self):
        self.beta_0 = 0  # Intercept
        self.beta_1 = 0  # Slope

    def fit(self, X, y):
        """
        Fit the model using the least squares method.
        """
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Calculate means
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        # Calculate beta_1 (slope)
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        self.beta_1 = numerator / denominator

        # Calculate beta_0 (intercept)
        self.beta_0 = y_mean - self.beta_1 * X_mean

    def predict(self, X):
        """
        Predict using the linear regression model.
        """
        X = np.array(X)
        return self.beta_0 + self.beta_1 * X

    def score(self, X, y):
        """
        Calculate the R^2 score of the model.
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


# Example usage
if __name__ == "__main__":
    # Sample data
    X = [23,45,67,35,-65]
    y = [2, 4, 5, 4, 5]

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict([1, 2, 3, 4, 5])

    # Print results
    print(f"Intercept (beta_0): {model.beta_0}")
    print(f"Slope (beta_1): {model.beta_1}")
    print(f"Predictions: {predictions}")
    print(f"R^2 score: {model.score(X, y)}")