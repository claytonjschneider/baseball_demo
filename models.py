import torch
from torch.autograd import Variable


class LinearRegressionModel(torch.nn.Module):
 
    def __init__(self, num_features=1, num_outcomes=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_outcomes)  # One in and one out
        self.double()

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    