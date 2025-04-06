from Logistic_Regression_with_a_Neural_Network_mindset.model_training.activation_function import sigmoid
import numpy as np
import copy
def propagate(x, y, w, b):
    m = x.shape[1]
    y_hat = sigmoid(np.dot(w.T, x) + b)
    epsilon = 1e-5  # 添加一个小的偏移量
    cost = - 1 / m * np.sum((y * np.log(y_hat + epsilon)) + ((1 - y) * np.log(1 - y_hat + epsilon)))
    dw = 1/m * np.dot(x, (y_hat - y).T)
    db = 1/m * np.sum(y_hat - y)
    cost = np.squeeze(np.array(cost))
    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost

def optimize(w, b, x, y, num_iterations=100, lr=0.01, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(x, y, w, b)
        dw = grads["dw"]
        db = grads["db"]
        w = w - lr * dw
        b = b - lr * db
        if i % 5 == 0:
            costs.append(cost)
            if print_cost:
                print(f"iteration: {i} cost: {cost}")
    grads = {
        "dw": dw,
        "db": db
    }
    params = {
        "w": w,
        "b": b
    }
    return grads, params, costs

if __name__ == "__main__":
    w = np.array([[1.], [2]])
    b = 1.5
    X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
    Y = np.array([[1, 1, 0]])
    grads, cost = propagate(X, Y, w, b)
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    print("cost = " + str(cost))
    grads, params, costs = optimize(w, b, X, Y)
    print("w = " + str(params["w"]))
    print("b = " + str(params["b"]))
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    print("Costs = " + str(costs))