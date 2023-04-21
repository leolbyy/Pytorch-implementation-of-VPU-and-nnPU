import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataset(data_path):
    data = np.loadtxt(os.path.join(data_path, 'pageblocks.txt'), delimiter=',')
    return data

def get_pageblocks_loaders(data_path, batch_size=200, num_labeled=400, positive_label_list=[1], test_proportion=0.3, val_proportion = 0.1):
    data_ndarray = get_dataset(data_path)

    positive = data_ndarray[np.isin(data_ndarray[:,-1], positive_label_list)]
    positive = np.concatenate((positive[:,:-1], np.ones(positive.shape[0]).reshape((-1, 1))), axis = 1)
    negative = data_ndarray[~np.isin(data_ndarray[:,-1], positive_label_list)]
    negative = np.concatenate((negative[:,:-1], np.zeros(negative.shape[0]).reshape((-1, 1))), axis = 1)

    pval_num, ptest_num = int(positive.shape[0] * val_proportion), int(positive.shape[0] * test_proportion)
    ptrain_num = positive.shape[0] - pval_num - ptest_num
    ptrain, pval, ptest = positive[:ptrain_num], positive[ptrain_num:pval_num + ptrain_num], positive[ptrain_num + pval_num:]

    nval_num, ntest_num = int(negative.shape[0] * val_proportion), int(negative.shape[0] * test_proportion)
    ntrain_num = negative.shape[0] - nval_num - ntest_num
    ntrain, nval, ntest = negative[:ntrain_num], negative[ntrain_num:nval_num + ntrain_num], negative[ntrain_num + nval_num:]

    
    train_num_labeled = num_labeled
    val_num_labeled = int(train_num_labeled * (val_proportion / (1 - val_proportion - test_proportion)))

    p_train, utrain = ptrain[:train_num_labeled], ptrain[train_num_labeled:]
    utrain = np.concatenate((utrain[:,:-1], np.zeros(utrain.shape[0]).reshape((-1, 1))), axis = 1)
    p_val, uval = pval[:val_num_labeled], pval[val_num_labeled:]
    uval = np.concatenate((uval[:, :-1], np.zeros(uval.shape[0]).reshape((-1, 1))), axis = 1)

    ptrain, pval = p_train, p_val
    utrain, uval = np.concatenate((utrain, ntrain), axis=0), np.concatenate((uval, nval), axis=0)
    train, val, test = np.concatenate((p_train, utrain, ntrain), axis=0), np.concatenate((p_val, uval, nval), axis=0), np.concatenate((ptest, ntest), axis=0)
    # train, val, test = train.astype(np.float32), val.astype(np.float32), test.astype(np.float32)
    xtrain, ytrain, xval, yval, xtest, ytest = train[:,:-1], train[:,-1], val[:,:-1], val[:,-1], test[:,:-1], test[:,-1]

    xptrain, yptrain, xutrain, yutrain = ptrain[:,:-1], ptrain[:,-1], utrain[:,:-1], utrain[:,-1]
    xpval, ypval, xuval, yuval = pval[:,:-1], pval[:,-1], uval[:,:-1], uval[:,-1]
    xtrain, ytrain, xval, yval, xtest, ytest = train[:,:-1], train[:,-1], val[:,:-1], val[:,-1], test[:,:-1], test[:,-1]

    xptrain, yptrain, xutrain, yutrain = xptrain.astype(np.float32), yptrain.astype(np.int32), xutrain.astype(np.float32), yutrain.astype(np.int32)
    xpval, ypval, xuval, yuval = xpval.astype(np.float32), ypval.astype(np.int32), xuval.astype(np.float32), yuval.astype(np.int32)
    xtrain, ytrain, xval, yval, xtest, ytest = xtrain.astype(np.float32), ytrain.astype(np.int32), xval.astype(np.float32), yval.astype(np.int32), xtest.astype(np.float32), ytest.astype(np.int32)
    

    # train_prior = p_train.shape[0] / train.shape[0]

    xptrain, yptrain, xutrain, yutrain = torch.from_numpy(xptrain), torch.from_numpy(yptrain), torch.from_numpy(xutrain), torch.from_numpy(yutrain)
    xpval, ypval, xuval, yuval = torch.from_numpy(xpval), torch.from_numpy(ypval), torch.from_numpy(xuval), torch.from_numpy(yuval)
    xtrain, ytrain, xval, yval, xtest, ytest = torch.from_numpy(xtrain), torch.from_numpy(ytrain), torch.from_numpy(xval), torch.from_numpy(yval), torch.from_numpy(xtest), torch.from_numpy(ytest)
    
    train_labeled, train_unlabeled = TensorDataset(xptrain, yptrain), TensorDataset(xutrain, yutrain)
    val_labeled, val_unlabeled = TensorDataset(xpval, ypval), TensorDataset(xuval, yuval)
    train, val, test = TensorDataset(xtrain, ytrain), TensorDataset(xval, yval), TensorDataset(xtest, ytest)


    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    train_labeled_loader = DataLoader(train_labeled, batch_size=batch_size, shuffle=True, drop_last=True)
    train_unlabeled_loader = DataLoader(train_unlabeled, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
    val_labeled_loader = DataLoader(val_labeled, batch_size=batch_size, shuffle=True)
    val_unlabeled_loader = DataLoader(val_unlabeled, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, train_labeled_loader, train_unlabeled_loader, val_loader, val_labeled_loader, val_unlabeled_loader, test_loader