import os
import torch
import numpy as np
import argparse
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from src.models import *
from src.loss_functions import *


def train(opt_phi, p_loader, x_loader):
    model.train()
    total_loss = 0
    for _ in range(config['val_iteration']):
        try:
            data_x, label_x = next(x_iter)
        except:
            x_iter = iter(x_loader)
            data_x, label_x = next(x_iter)
        try:
            data_p, label_p = next(p_iter)
        except:
            p_iter = iter(p_loader)
            data_p, label_p = next(p_iter)

        data_p, data_x, label_p, label_x = data_p.to(device), data_x.to(device), label_p.to(device), label_x.to(device)
        data_all = torch.cat((data_p, data_x))
        pred_all = model(data_all)
        loss = loss_func(pred_all, torch.cat((label_p, label_x)))

        if method == 'vpu':
            # perform additional mix-up step
            target_unlabeled = pred_all[len(data_p):, 1].exp()
            target_posi = torch.ones(len(data_p), dtype=torch.float32, device=device)
            lam = torch.distributions.beta.Beta(config['mix_alpha'], config['mix_alpha']).sample()
            data = lam * data_x + (1 - lam) * data_p
            target = lam * target_unlabeled + (1 - lam) * target_posi
            data, target = data.to(device), target.to(device)
            mixed_pred_all = model(data)
            reg_mix_log = ((torch.log(target) - mixed_pred_all[:, 1]) ** 2).mean()
            loss = loss + config['lam'] * reg_mix_log

        total_loss += loss.item()

        opt_phi.zero_grad()
        loss.backward()
        opt_phi.step()

    return total_loss / config['val_iteration']


def evaluate(test_loader, epoch, train_loss, val_loss):
    model.eval()

    pred_all, target_all = None, None

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_all = output if pred_all is None else torch.cat((pred_all, output))
            target_all = target if target_all is None else torch.cat((target_all, target))
    pred_all = pred_all[:,1]
    pred_all = np.array(pred_all.cpu().detach())
    pred_all_class = np.array(pred_all > np.math.log(0.5))
    target_all = np.array(target_all.cpu().detach())
    test_acc = accuracy_score(target_all, pred_all_class)
    test_auc = roc_auc_score(target_all, pred_all)
    # print(f'classification report: \n{classification_report(target_all, pred_all_class)}', end = '\n\n')
    print('Train Epoch: {}\t train_loss: {:.4f}  Test accuracy: {:.4f}   Test AUC : {:.4f}    Val loss: {:.4f}' \
        .format(epoch, train_loss, test_acc, test_auc, val_loss))
    return test_acc, test_auc


def cal_val_loss(val_p, val_x):
    model.eval()
    data_all, target_all = None, None

    with torch.no_grad():
        for data, target in val_p:
            data, target = data.to(device), target.to(device)
            data_all = data if data_all is None else torch.cat((data_all, data))
            target_all = target if target_all is None else torch.cat((target_all, target))
        
        for data, target in val_x:
            data, target = data.to(device), target.to(device)
            data_all = data if data_all is None else torch.cat((data_all, data))
            target_all = target if target_all is None else torch.cat((target_all, target))
        
        pred_all = model(data_all)
        val_loss = loss_func(pred_all, target_all)
    return val_loss
    

    
        
        

def run(loaders):
    global device
    global model
    global loss_func
    global method
    device = config['device']
    model = config['model']
    loss_func = config['loss_func']
    method = config['method']

    lowest_val_loss = float('inf')

    (train_loader, train_labeled_loader, train_unlabeled_loader, val_loader, val_labeled_loader, val_unlabeled_loader, test_loader) = loaders

    learning_rate = config['learning_rate'] * 2
    opt = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.5, 0.99))

    for epoch in range(config['epochs']):
        if epoch % 20 == 0:
            learning_rate /= 2
            opt = torch.optim.Adam(model.parameters(), lr = learning_rate, betas= (0.5, 0.99))
        train_loss = train(opt, train_labeled_loader, train_unlabeled_loader)
        val_loss = cal_val_loss(val_labeled_loader, val_unlabeled_loader)
        test_acc, test_auc = evaluate(test_loader, epoch, train_loss, val_loss)

        if val_loss < lowest_val_loss:
            best_epoch = epoch + 1
            best_val_loss = val_loss
            best_test_acc = test_acc
            best_test_auc = test_auc
            lowest_val_loss = val_loss
        
    print(f'Early stopping at {best_epoch} epoch with val_loss: {best_val_loss}, test_acc: {best_test_acc}, test_auc: {best_test_auc}')
    return best_test_acc


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSA5204 Group 12 reproduction of VPU')
    parser.add_argument('--dataset', default='fashionMNIST',
                        choices=['fashionMNIST', 'pageblocks'])
    parser.add_argument('--method', default='vpu', choices=['vpu', 'nnpu', 'raw'])
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--data_path', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'data')), help='path for storing raw dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to run on. Only valid when running on cude available machines')
    parser.add_argument('--prior', type=float, default=0.3, help='prior for nnpu method')
    config = vars(parser.parse_args())

    # global config:
    if torch.backends.mps.is_available():
        config['device'] = torch.device('mps')
    elif torch.cuda.is_available():
        config['device'] = torch.device(f"cuda:{config['gpu']}")
    else:
        config['device'] = torch.device('cpu')
    config['learning_rate'] = 3e-5
    config['val_iteration'] = 30
    config['epochs'] = 50

    # config for methods
    if config['method'] == 'vpu':
        config['mix_alpha'] = 0.3
        config['lam'] = 0.03
        config['loss_func'] = vpu()
    elif config['method'] == 'nnpu':
        config['loss_func'] = nnPU(prior = config['prior'])
    else:
        config['loss_func'] = raw()

    # config for dataset
    if config['dataset'] == 'fashionMNIST':
        config['model'] = Conv().to(config['device'])
        config['num_labeled'] = 3000
        config['positive_label_list'] = [1, 4, 7]
        from src.data.FashionMNIST import getFashionMNISTLoader as get_loaders
    else:
        config['model'] = MLP().to(config['device'])
        config['num_labeled'] = 400
        config['positive_label_list'] = [2,3,4,5]
        from src.data.PageBlocks import get_pageblocks_loaders as get_loaders


    train_loader, train_labeled_loader, train_unlabeled_loader, \
        val_loader, val_labeled_loader, val_unlabeled_loader, \
            test_loader = get_loaders(data_path=config['data_path'], 
                                    batch_size=config['batch_size'], 
                                    num_labeled=config['num_labeled'], 
                                    positive_label_list=config['positive_label_list'])
    loaders = (train_loader, train_labeled_loader, train_unlabeled_loader, val_loader, val_labeled_loader, val_unlabeled_loader, test_loader)

    print('==> Preparing data')
    print('    # train data: ', len(train_loader.dataset))
    print('    # labeled train data: ', len(train_labeled_loader.dataset))
    print('    # test data: ', len(test_loader.dataset))
    print('    # validation data:', len(val_loader.dataset))
    print('    # labeled validation data:', len(val_labeled_loader.dataset))

    run(loaders)

