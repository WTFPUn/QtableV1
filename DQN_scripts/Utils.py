import math
import random
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.normal import Normal
from collections import deque

# create a deque object to represent the queue
queue = deque()
np.random.seed(690)
##############################################################################
# if gpu is to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
ALPHA = 0.8
BATCH_SIZE = 2
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TAU = 0.005
LR = 1e-4

##############################################################################

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        if batch_size > len(self.memory):
            raise ValueError("batch_size must be less than or equal to the memory size")
        indices = torch.randint(low=0, high=len(self.memory), size=(batch_size,), dtype=torch.long)
        batch = [self.memory[i] for i in indices]
        return batch

    def __len__(self):
        return len(self.memory)

def sigmoid(x):
    return 1./(1. + np.exp(-x)) 

def catergory(x):
    return int(1) if sigmoid(x) > 0.5 else int(0)

def apply_func(tensor):
    tensor[:, 0].detach().apply_(catergory)
    tensor[:, 1] = ALPHA*tensor[:, 1].detach().apply_(sigmoid)
    tensor[:, 2] = (np.pi/4.)*tensor[:, 2]
    return Variable(tensor.type(torch.float64), requires_grad = True)

def select_action(state, steps_done, policy_net):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if torch.rand(1) > eps_threshold:
        print(f'get Best Action')
        action_status = 'get Best Action'
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #return policy_net(state).max(1)[1].view(1, 1), eps_threshold
            # pred = policy_net(state)
            # print(f'pred: {pred}  {type(pred)}, {pred.size()}')
            return policy_net(state)[0], action_status
    else:
        # return torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)
        print(f'get Random Action')
        action_status = 'get Random Action'
        #return torch.tensor(np.random.randint(5, size = 1), device=DEVICE, dtype=torch.long), eps_threshold
        # return torch.tensor(np.random.uniform(0.0, 1.0, size=2), device=DEVICE, dtype=torch.long), eps_threshold
        return torch.tensor([np.random.randint(2), torch.rand(1), 2*torch.rand(1) - 1], device=DEVICE).view(3), action_status
        # return torch.tensor([random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)], device=DEVICE, dtype=torch.long), eps_threshold


def optimize_model(policy_net, target_net, optimizer, memory = ReplayMemory(10000), criterion = nn.SmoothL1Loss()):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=DEVICE, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    # print(f'state_batch: size {state_batch.size()}  {state_batch}')
    action_batch = torch.cat(batch.action).view(BATCH_SIZE, 3)
    # print(f'action_batch: size {action_batch.size()}  {action_batch}')
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    tensor = policy_net(state_batch)
    with torch.no_grad():
        tensor[:, 0].detach().apply_(catergory)
        tensor[:, 1] = ALPHA*tensor[:, 1].detach().apply_(sigmoid)
        tensor[:, 2] = (np.pi/4.)*tensor[:, 2].detach().apply_(np.tanh)

    state_action_values =  Variable(tensor.type(torch.float64), requires_grad = True)
    # print(f'state_action_values: size {state_action_values.size()}  {state_action_values}')
    state_action_values = state_action_values.gather( 1, action_batch.long())
    # print('state_action_values: ', state_action_values.size())

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA ) + reward_batch

    # Compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss