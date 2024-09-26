import numpy as np

# Get input from the user
x_input = input("Enter the value of X (comma-separated): ")
y_input = input("Enter the value of Y (comma-separated): ")

# Convert input strings to lists of floats
x = list(map(float, x_input.split(',')))
y = list(map(float, y_input.split(',')))

# Define the Linearregression class
class Linearregression:
    def fit(self, x, y):
        self.arrayx = np.array(x)
        self.xm = np.mean(self.arrayx)
        self.arrayy = np.array(y)
        self.ym = np.mean(self.arrayy)
        self.numerator = np.sum((self.arrayx - self.xm) * (self.arrayy - self.ym))
        self.denominator = np.sum((self.arrayx - self.xm) ** 2)
        self.b1 = self.numerator / self.denominator
        self.b0 = self.ym - (self.b1 * self.xm)
        return self.b0, self.b1

    def predict(self, x):
        ypre = self.b0 + (self.b1 * np.array(x))
        return ypre

    def residual(self, ypre, y):
        residuals = np.sum((np.array(y) - np.array(ypre)))**2
        return residuals

    def tss(self , y):
        tss = np.sum(y - np.mean(y))**2


# Create an instance of the Linearregression class and fit the model
model = Linearregression()
b0, b1 = model.fit(x, y)

# Print the coefficients
print(f"b0: {b0}")
print(f"b1: {b1}")

Linearregression.tss()
