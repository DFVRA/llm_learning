from Logistic_Regression_with_a_Neural_Network_mindset.model_training.activation_function import sigmoid
import numpy as np

def predict(params, x):
    w = params["w"]
    b = params["b"]
    y_hat = sigmoid(np.dot(w.T, x) + b)
    print(y_hat.shape)
    y_hat = (y_hat >= 0.5).astype(int).squeeze()
    return y_hat


if __name__ == "__main__":
    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    params = {
        "w": w,
        "b": b
    }
    X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    print("predictions = " + str(predict(params, X)))
