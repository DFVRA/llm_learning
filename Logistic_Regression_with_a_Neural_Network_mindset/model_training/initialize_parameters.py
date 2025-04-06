import numpy as np
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    parameters = {
        "w": w,
        "b": b
    }
    return parameters

if __name__ == "__main__":
    parameters = initialize_with_zeros(5)
    print(f"initialize w shape:{parameters['w'].shape}")