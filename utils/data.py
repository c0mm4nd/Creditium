import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.datasets import cifar10, cifar100, mnist


def load_MNIST_data(standarized=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

    return X_train, y_train, X_test, y_test


def load_EMNIST_data(file, standarized=False):
    """
    file should be the downloaded EMNIST file in .mat format.
    """
    import scipy

    mat = scipy.io.loadmat(file)
    data = mat["dataset"]

    writer_ids_train = data["train"][0, 0]["writers"][0, 0]
    writer_ids_train = np.squeeze(writer_ids_train)
    X_train = data["train"][0, 0]["images"][0, 0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order="F")
    y_train = data["train"][0, 0]["labels"][0, 0]
    y_train = np.squeeze(y_train)
    y_train -= 1  # y_train is zero-based

    writer_ids_test = data["test"][0, 0]["writers"][0, 0]
    writer_ids_test = np.squeeze(writer_ids_test)
    X_test = data["test"][0, 0]["images"][0, 0]
    X_test = X_test.reshape((X_test.shape[0], 28, 28), order="F")
    y_test = data["test"][0, 0]["labels"][0, 0]
    y_test = np.squeeze(y_test)
    y_test -= 1  # y_test is zero-based

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

    return X_train, y_train, X_test, y_test, writer_ids_train, writer_ids_test


def generate_partial_data(X, y, class_in_use=None):
    if class_in_use is None:
        idx = np.ones_like(y, dtype=bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis=0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    return X_incomplete, y_incomplete


def generate_bal_private_data(
    X,
    y,
    N_parties=10,
    classes_in_use=list(range(11)),
    N_samples_per_class=20,
    data_overlap=False,
):
    """
    Input:
    -- N_parties : int, number of collaboraters in this activity;
    -- classes_in_use: array or generator, the classes of EMNIST-letters dataset
    (0 <= y <= 25) to be used as private data;
    -- N_sample_per_class: int, the number of private data points of each class for each party

    return:

    """
    priv_data = []
    combined_idx = np.array([], dtype=np.int16)
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        idx = np.random.choice(
            idx, N_samples_per_class * N_parties, replace=data_overlap
        )
        combined_idx = np.r_[combined_idx, idx]
        for i in range(N_parties):
            idx_tmp = idx[i * N_samples_per_class : (i + 1) * N_samples_per_class]
            if i == len(priv_data):
                # first time
                tmp = {}
                tmp["X"] = X[idx_tmp]
                tmp["y"] = y[idx_tmp]
                tmp["idx"] = idx_tmp
                priv_data.append(tmp)
            else:
                priv_data[i]["idx"] = np.r_[priv_data[i]["idx"], idx_tmp]
                priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]

    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return priv_data, total_priv_data


def generate_EMNIST_writer_based_data(
    X, y, writer_info, N_priv_data_min=30, N_parties=5, classes_in_use=range(6)
):
    # mask is a boolean array of the same shape as y
    # mask[i] = True if y[i] in classes_in_use
    mask = None
    mask = [y == i for i in classes_in_use]
    mask = np.any(mask, axis=0)

    df_tmp = None
    df_tmp = pd.DataFrame({"writer_ids": writer_info, "is_in_use": mask})
    # print(df_tmp.head())
    groupped = df_tmp[df_tmp["is_in_use"]].groupby("writer_ids")

    # organize the input the data (X,y) by writer_ids.
    # That is,
    # data_by_writer is a dictionary where the keys are writer_ids,
    # and the contents are the correcponding data.
    # Notice that only data with labels in class_in_use are included.
    data_by_writer = {}
    writer_ids = []
    for wt_id, idx in groupped.groups.items():
        if len(idx) >= N_priv_data_min:
            writer_ids.append(wt_id)
            data_by_writer[wt_id] = {
                "X": X[idx],
                "y": y[idx],
                "idx": idx,
                "writer_id": wt_id,
            }

    # each participant in the collaborative group is assigned data
    # from a single writer.
    ids_to_use = np.random.choice(writer_ids, size=N_parties, replace=False)
    combined_idx = np.array([], dtype=np.int64)
    private_data = []
    for i in range(N_parties):
        id_tmp = ids_to_use[i]
        private_data.append(data_by_writer[id_tmp])
        combined_idx = np.r_[combined_idx, data_by_writer[id_tmp]["idx"]]
        del id_tmp

    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return private_data, total_priv_data


def generate_alignment_data(X, y, N_alignment=3000):
    split = StratifiedShuffleSplit(n_splits=1, train_size=N_alignment)
    if N_alignment == "all":
        alignment_data = {}
        alignment_data["idx"] = np.arange(y.shape[0])
        alignment_data["X"] = X
        alignment_data["y"] = y
        return alignment_data
    for train_index, _ in split.split(X, y):
        X_alignment = X[train_index]
        y_alignment = y[train_index]
    alignment_data = {}
    alignment_data["idx"] = train_index
    alignment_data["X"] = X_alignment
    alignment_data["y"] = y_alignment

    return alignment_data
