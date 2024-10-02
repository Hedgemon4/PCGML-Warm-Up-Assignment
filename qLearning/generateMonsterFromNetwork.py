import torch

from qTabular.MDP import (
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

from qNetwork import QNetwork

# Hyperparameters
file_name = "policy_net.pth"
maxAttempts = 10  # max number of attempts allowed to find a balanced agent
maxRolloutLength = (
    100  # Make number of changes to attempt to make to find a balanced monster
)

# Load network
model = QNetwork()
model.load_state_dict(torch.load(file_name))
model.eval()

# Whether or not we're done
done = False
rolloutIndex = 0

# Tracking the best monster we've found so far
bestMonster = None
bestReward = -1

for attempt in range(0, maxAttempts):
    monsterInit = RandomState()
    state = monsterInit.clone()

    while not done and rolloutIndex < maxRolloutLength:
        rolloutIndex += 1

        state_tensor = state.get_tensor()
        action_tensor = model(state_tensor)
        maxValue = torch.max(action_tensor)
        print(maxValue)
        action = action_tensor.argmax().item()

        nextState = state.clone()

        if action == 0:
            nextState = RaiseHealth(nextState)
        elif action == 1:
            nextState = LowerHealth(nextState)
        elif action == 2:
            nextState = RaiseArmor(nextState)
        elif action == 3:
            nextState = LowerArmor(nextState)
        elif action == 4:
            nextState = RaiseSpeed(nextState)
        elif action == 5:
            nextState = LowerSpeed(nextState)
        elif action == 6:
            nextState = RaiseDamage(nextState)
        elif action == 7:
            nextState = LowerDamage(nextState)

        state = nextState

    reward = CalculateReward(state)

    if reward > bestReward:
        bestMonster = state
        bestReward = reward

# Print the final monster and reward value
print("Final Reward Value: " + str(bestReward))
print("Final Monster: " + str(bestMonster))
