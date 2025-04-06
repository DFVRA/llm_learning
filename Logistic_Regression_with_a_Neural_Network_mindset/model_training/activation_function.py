import numpy as np

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

if __name__ == "__main__":
    z = np.array([0, 2])
    s = sigmoid(z)
    print(f"{s[0], s[1]}")