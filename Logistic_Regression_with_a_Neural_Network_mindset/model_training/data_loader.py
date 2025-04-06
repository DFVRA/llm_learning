import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('./datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def standard_data_set(dataset):
    train_set_x_origin = dataset["train_set_x_origin"]
    train_set_y_origin = dataset["train_set_y_origin"]
    test_set_x_origin = dataset["test_set_x_origin"]
    test_set_y_origin = dataset["test_set_y_origin"]
    train_set_x = train_set_x_origin.reshape(train_set_x_origin.shape[0], -1).T / 255
    test_set_x = test_set_x_origin.reshape(test_set_x_origin.shape[0], -1).T / 255
    return {
        "train_set_x": train_set_x,
        "train_set_y": train_set_y_origin,
        "test_set_x": test_set_x,
        "test_set_y": test_set_y_origin
    }




if __name__ == "__main__":
    train_set_x_origin, train_set_y_origin, test_set_x_origin, test_set_y_origin, classes = load_dataset()
    print(f"Before standard:")
    print(f"train_set_x shape:{train_set_x_origin.shape}")
    print(f"train_set_y shape:{train_set_y_origin.shape}")
    print(f"test_set_x shape:{test_set_x_origin.shape}")
    print(f"test_set_y shape:{test_set_y_origin.shape}")
    dataset_before_standard = {
        "train_set_x_origin": train_set_x_origin,
        "train_set_y_origin": train_set_y_origin,
        "test_set_x_origin": test_set_x_origin,
        "test_set_y_origin": test_set_y_origin
    }

    dataset_after_standard = standard_data_set(dataset_before_standard)
    train_set_x = dataset_after_standard["train_set_x"]
    train_set_y = dataset_after_standard["train_set_y"]
    test_set_x = dataset_after_standard["test_set_x"]
    test_set_y = dataset_after_standard["test_set_y"]
    print(f"After standard:")
    print(f"train_set_x shape:{train_set_x.shape}")
    print(f"train_set_y shape:{train_set_y.shape}")
    print(f"test_set_x shape:{test_set_x.shape}")
    print(f"test_set_y shape:{test_set_y.shape}")
