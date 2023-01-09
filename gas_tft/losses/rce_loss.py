import torch


class RCELoss(torch.nn.Module):
    def __init__(self, mean, std, device):
        super(RCELoss, self).__init__()
        self.mean = mean
        self.std = std
        self.device = device

    def forward(self, predictions, actuals):
        predictions = predictions.float() * self.std + self.mean
        actuals = actuals.float() * self.std + self.mean
        sumf = torch.abs(torch.sum(predictions, dim=1) - torch.sum(actuals, dim=1)) / torch.sum(actuals, dim=1)

        return torch.mean(sumf)
