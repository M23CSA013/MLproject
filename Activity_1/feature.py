import numpy as np

def sigmoid(x):
   return 1 / (1 + np.exp(-x))
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

random_values = np.array(random_values)

sigmoid_values = sigmoid(random_values)

print("Random Values:")
for value in random_values:
   print("{:.2f}".format(value), end=", ")
print()

print("Sigmoid Values:")
for value in sigmoid_values:
   print("{:.4f}".format(value), end=", ")
print()