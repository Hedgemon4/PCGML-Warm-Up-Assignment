import math
import random
from itertools import count

import torch
from torch import nn

from MDP import (
    RandomState,
    RaiseHealth,
    LowerHealth,
    RaiseArmor,
    LowerArmor,
    RaiseSpeed,
    LowerSpeed,
    RaiseDamage,
    LowerDamage,
    CalculateReward,
)
from ReplayMemory import ReplayMemory, Transition
from qNetwork import QNetwork
import torch.optim as optim


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
NUM_EPISODES = 50
MAX_ROLLOUT_LENGTH = 50

n_actions = 8
n_observations = 5

policy_network = QNetwork()
target_network = QNetwork()
target_network.load_state_dict(policy_network.state_dict())

optimizer = optim.AdamW(policy_network.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def sample_action():
    return random.randint(0, 4)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            item = policy_network(state).argmax().long()
            return torch.tensor([[item]])
    else:
        return torch.tensor([[sample_action()]], dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transactions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transactions))

    # state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    # reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.stack(batch.next_state)
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # state_batch = torch.stack([t[0] for t in state_batch])

    state_action_values = policy_network(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_state_values = target_network(next_state_batch).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optimizer.step()


for i in range(NUM_EPISODES):
    print("Episode :" + str(i))
    monsterInit = RandomState()

    state = (
        monsterInit.clone()
    )  # Clone the state to make sure we can make changes to it and still track it
    rollout_index = 0

    while rollout_index < MAX_ROLLOUT_LENGTH:
        state_tensor = state.get_tensor()
        action = select_action(state_tensor)

        nextState = state.clone()
        action_value = action.item()

        if action_value == 0:
            nextState = RaiseHealth(nextState)
        elif action_value == 1:
            nextState = LowerHealth(nextState)
        elif action_value == 2:
            nextState = RaiseArmor(nextState)
        elif action_value == 3:
            nextState = LowerArmor(nextState)
        elif action_value == 4:
            nextState = RaiseSpeed(nextState)
        elif action_value == 5:
            nextState = LowerSpeed(nextState)
        elif action_value == 6:
            nextState = RaiseDamage(nextState)
        elif action_value == 7:
            nextState = LowerDamage(nextState)

        reward = CalculateReward(nextState)
        reward = torch.tensor([reward], dtype=torch.float)

        next_state_tensor = nextState.get_tensor()

        # Store result
        memory.push(state_tensor, action, reward, next_state_tensor)

        optimize_model()

        target_net_state_dict = target_network.state_dict()
        policy_net_state_dict = policy_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_network.load_state_dict(target_net_state_dict)
        rollout_index += 1
