import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

# Set to GPU mode is nvidia-gpu is available
# Can change to MPS is you are using MacOS
# if torch.cuda.is_available():
    # torch.cuda.set_device()

# Download and preprocess dataset
def get_dataset(data_path):
    train, test = torchvision.datasets.FashionMNIST(data_path, train=True, download=True), torchvision.datasets.FashionMNIST(data_path, train=False, download=True)
    (xtrain, ytrain), (xtest, ytest) = (train.data, train.targets), (test.data, test.targets)
    xtrain, xtest = xtrain.type(torch.float32), xtest.type(torch.float32)
    xtrain, xtest = xtrain / 255. , xtest / 255.
    # reshape for model input
    xtrain = xtrain.reshape(xtrain.shape[0], 1, xtrain.shape[1], xtrain.shape[2])
    xtest = xtest.reshape(xtest.shape[0], 1, xtest.shape[1], xtest.shape[2])

    return (xtrain, ytrain), (xtest, ytest)


# Transform to binary classification data and split train and validation
def train_val_split(X, Y, positive_label_list = [1, 4, 7], num_labeled = 3000):
    assert X.shape[0] == Y.shape[0], "number of sample and number of label is not the same"

    train_num_per_class = 5500
    train_num_labeled_per_class = int(num_labeled / len(positive_label_list))
    val_num_per_class = 500
    val_num_labeled_per_class = int(val_num_per_class * (train_num_labeled_per_class / 6000))

    train_labeled_idx = []
    train_unlabeled_idx = []
    val_labeled_idx = []
    val_unlabeled_idx = []


    for i in range(10): # 10 labels:
        idx = np.where(Y == i)[0]
        np.random.shuffle(idx)

        if i in positive_label_list:
            train_labeled_idx.extend(idx[:train_num_labeled_per_class])
            train_unlabeled_idx.extend(idx[:train_num_per_class])
            val_labeled_idx.extend(idx[-val_num_labeled_per_class:])
            val_unlabeled_idx.extend(idx[-val_num_per_class:])
        else:
            train_unlabeled_idx.extend(idx[:train_num_per_class])
            val_unlabeled_idx.extend(idx[-val_num_per_class:])
    
    return train_labeled_idx, train_unlabeled_idx, val_labeled_idx, val_unlabeled_idx






def getFashionMNISTLoader(data_path, positive_label_list=[1, 4, 7], batch_size=500, num_labeled=3000):
    (xtrain, ytrain), (xtest, ytest) = get_dataset(data_path)
    train_labeled_idx, train_unlabeled_idx, val_labeled_idx, val_unlabeled_idx = train_val_split(xtrain, ytrain, positive_label_list, num_labeled)

    xtrain_labeled, ytrain_labeled = xtrain[train_labeled_idx], torch.ones((len(train_labeled_idx),), dtype=torch.int64)
    xtrain_unlabeled, ytrain_unlabeled = xtrain[train_unlabeled_idx], torch.zeros((len(train_unlabeled_idx),), dtype=torch.int64)

    xval_labeled, yval_labeled = xtrain[val_labeled_idx], torch.ones((len(val_labeled_idx),), dtype=torch.int64)
    xval_unlabeled, yval_unlabeled = xtrain[val_unlabeled_idx], torch.zeros((len(val_unlabeled_idx),), dtype=torch.int64)

    xtrain, ytrain = torch.cat([xtrain_labeled, xtrain_unlabeled], axis=0), torch.cat([ytrain_labeled, ytrain_unlabeled], axis=0)
    xval, yval = torch.cat([xval_labeled, xval_unlabeled], axis=0), torch.cat([yval_labeled, yval_unlabeled], axis=0)

    ytest = torch.isin(ytest, torch.tensor(positive_label_list)).type(torch.int64)
    
    train, train_labeled, train_unlabeled = TensorDataset(xtrain, ytrain), TensorDataset(xtrain_labeled, ytrain_labeled), TensorDataset(xtrain_unlabeled, ytrain_unlabeled)
    val, val_labeled, val_unlabeled = TensorDataset(xval, yval), TensorDataset(xval_labeled, yval_labeled), TensorDataset(xval_unlabeled, yval_unlabeled)
    test = TensorDataset(xtest, ytest)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    train_labeled_loader = DataLoader(train_labeled, batch_size=batch_size, shuffle=True, drop_last=True)
    train_unlabeled_loader = DataLoader(train_unlabeled, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
    val_labeled_loader = DataLoader(val_labeled, batch_size=batch_size, shuffle=True)
    val_unlabeled_loader = DataLoader(val_unlabeled, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, train_labeled_loader, train_unlabeled_loader, val_loader, val_labeled_loader, val_unlabeled_loader, test_loader