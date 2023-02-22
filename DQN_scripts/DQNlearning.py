import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions=2 ):
        super(DQN, self).__init__()
        self.Input = nn.Linear(n_observations, 128)
        self.layer1 = nn.Linear(128, 32)
        self.layer2 = nn.Linear(32, 8)
        self.Output = nn.Linear(8, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.Input(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.tanh(self.Output(x)) # [-1, 1]
    