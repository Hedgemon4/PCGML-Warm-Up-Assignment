import csv

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
function_name = "policy_net_max_points_125_total_episodes_100"
output_folder = "generated_content/policy_network_125_points/"
file_name = "networks/" + function_name + ".pth"
maxAttempts = 10  # max number of attempts allowed to find a balanced agent
maxRolloutLength = (
    100  # Make number of changes to attempt to make to find a balanced monster
)

# Load network
model = QNetwork()
model.load_state_dict(torch.load(file_name))
model.eval()

num_runs = 20

for i in range(num_runs):
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

    output_file_name = output_folder + function_name + "_monster_number_" + str(i)
    with open(output_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Final Reward Value', 'Final Monster'])
        writer.writerow([bestReward, bestMonster])

    # Print the final monster and reward value
    print("Final Reward Value: " + str(bestReward))
    print("Final Monster: " + str(bestMonster))
