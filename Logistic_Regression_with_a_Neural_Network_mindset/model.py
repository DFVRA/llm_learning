from Logistic_Regression_with_a_Neural_Network_mindset.model_training.data_loader import load_dataset, standard_data_set
from Logistic_Regression_with_a_Neural_Network_mindset.model_training.initialize_parameters import initialize_with_zeros
from Logistic_Regression_with_a_Neural_Network_mindset.model_training.predict import predict
from Logistic_Regression_with_a_Neural_Network_mindset.model_training.propagate import optimize
import numpy as np

def model(x_train, y_train, x_test, y_test, num_iterations=300, lr=0.5, print_cost=True):
    m = x_train.shape[1]
    dim = x_train.shape[0]
    params = initialize_with_zeros(dim)
    w = params["w"]
    b = params["b"]
    grads, params, costs = optimize(w, b, x_train, y_train, num_iterations, lr, print_cost)
    y_train_pre = predict(params, x_train)
    y_test_pre = predict(params, x_test)
    if print_cost:
        print(f"train accuracy:{100 - np.mean(np.abs(y_train_pre - y_train)) * 100}%")
        print(f"test accuracy:{100 - np.mean(np.abs(y_test_pre - y_test)) * 100}%")
    d = {
            "costs": costs,
             "Y_prediction_test": y_test_pre,
             "Y_prediction_train": y_train_pre,
             "w": w,
             "b": b,
             "learning_rate": lr,
             "num_iterations": num_iterations
        }

    return d


if __name__ == "__main__":
    train_set_x_origin, train_set_y_origin, test_set_x_origin, test_set_y_origin, classes = load_dataset()
    dataset_before_standard = {
        "train_set_x_origin": train_set_x_origin,
        "train_set_y_origin": train_set_y_origin,
        "test_set_x_origin": test_set_x_origin,
        "test_set_y_origin": test_set_y_origin
    }

    dataset_after_standard = standard_data_set(dataset_before_standard)
    x_train = dataset_after_standard["train_set_x"]
    y_train = dataset_after_standard["train_set_y"]
    x_test = dataset_after_standard["test_set_x"]
    y_test = dataset_after_standard["test_set_y"]
    d = model(x_train, y_train, x_test, y_test)
    w = d["w"]
    b = d["b"]
