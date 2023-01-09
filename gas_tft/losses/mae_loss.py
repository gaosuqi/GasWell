import torch


class MAELoss(torch.nn.Module):
    def __init__(self, mean, std, device):
        super(MAELoss, self).__init__()
        self.mean = mean
        self.std = std
        self.device = device

    def forward(self, predictions, actuals):
        sequence_length = predictions.shape[1]
        predictions = predictions.float() * self.std + self.mean
        actuals = actuals.float() * self.std + self.mean
        sumf = torch.sum(torch.abs(predictions - actuals), dim=1)

        return torch.mean(sumf / sequence_length)
