import numpy as np

def relu(y):
   # Correcting the ReLU implementation to handle negative values properly
   return np.maximum(0, y)

def leaky_relu(x, alpha=0.01):
   return np.where(np.array(x) > 0, x, alpha*x)

def tanh(x):
   return np.tanh(x)

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

random_values = np.array(random_values)

relu_values = relu(random_values)

leaky_relu_values = leaky_relu(random_values)

tanh_values = tanh(random_values)

print("Random Values:")
for value in random_values:
   print("{:.2f}".format(value), end=", ")
print()

print("Random Values:", random_values)
print("ReLU Values:", relu_values)
print("Leaky ReLU Values:", leaky_relu_values)
print("Tanh Values:", tanh_values)

