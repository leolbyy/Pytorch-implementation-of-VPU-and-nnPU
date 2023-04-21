import torch


class nnPU():
    def __init__(self, prior) -> None:
        self.prior = prior
        # self.device = device

    def __call__(self, pred, target):
        prior = torch.tensor(self.prior, dtype = torch.float32)
        min_count = torch.tensor(1., dtype = torch.float32)
        # prior = prior.to(self.device)
        # min_count = min_count.to(self.device)

        pred = pred[:, 1]
        pred = pred.exp()

        positive, unlabeled = target == 1, target == 0
        positive, unlabeled = positive.type(torch.float32), unlabeled.type(torch.float32)

        n_positive, n_unlabeled = torch.max(min_count, torch.sum(positive)), torch.max(min_count, torch.sum(unlabeled))

        y_positive = torch.sigmoid(-pred) * positive
        y_inv = torch.sigmoid(pred) * positive
        y_unlabeled = torch.sigmoid(pred) * unlabeled

        positive_risk = prior * torch.sum(y_positive) / n_positive
        negative_risk = -prior * torch.sum(y_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < 0:
            return -negative_risk
        else:
            return positive_risk + negative_risk
        

class raw():
    def __init__(self) -> None:
        self.loss_func = torch.nn.CrossEntropyLoss()
    def __call__(self, pred, target):
        return self.loss_func(pred, target)


class vpu():
    def __init__(self) -> None:
        pass

    def __call__(self, pred, target):
        pred_all = pred[:, 1]
        num_posi = torch.sum(target == 1)
        pred_posi = pred_all[:num_posi]
        pred_unalabeled = pred_all[num_posi:]
        var_loss = torch.logsumexp(pred_unalabeled, dim=0) - torch.log(num_posi) - torch.mean(pred_posi)
        return var_loss
