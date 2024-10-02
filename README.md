# PCGML WarmUp Assignment - Balanced Monster Generation

## Running Modified Tabular Code

- The original code for the example is located in the original code folder.
- The [qTabular](qTabular) folder contains the original code with some modifications. It can be run the same way as the original code.
- These modifications include:
  - Changes to the hyperparameters.
  - Implementing a max points system, so the points of our generated monster can only go so high.
  - The monster now does not only fight the balanced monster, but all four monsters mentioned in the textbook.

## Running Modified Q-Learning Code

- I also modified the original code to implement DQN. Instead of using a tabular Q, we use neural networks in PyTorch to approximate the Q function.
- In addition to pickle, some other packages are needed. These can be installed with pip in a virtual environment with the following command `python -m pip install -r requirements.txt`.
- Some packages may be platform dependent (I did this on a Mac). If the code does not run after installing the packages, please use the commands that PyTorch gives on [this](https://pytorch.org/get-started/locally/) page to install PyTorch for your system. You will not need pickle for this portion. 
- At the top of the [Generate Monster](qLearning/generateMonsterFromNetwork.py) file, there is a variable to change the trained network being used to generate the content. Changing that line to any of the networks I included in the [networks](networks) folder will allow you to generate a monster using that network. The generation works the same as before, except that it uses the model to choose what action to take, and I removed the `doneThreshold` condition because my network does not produce values in the same range as the tabular version.

## Old Instructions

Scripts: 
- MDP.py defines the Markov Decision Process based on the one from Chapter 10
- Qlearner.py defines the Q-learning training environment 
- GenerateMonster.py queries the trained Q-learning agent to generate a 'balanced' monster

Instructions: 
1. Install Python 3.9 (but tweaking it for other versions should be simple) and the 'pickle' library
2. Run GenerateMonster.py to generate a 'balanced' monster according to the reward function defined in MDP.py and based on a pretrained Q-learner
3. Alter the hyperparameters at the top of Qlearner.py to train a new Q-learner agent
4. Swap the input file at the top of GenerateMonster.py to use your new Q-learner agent and get output from it
5. Make further changes to GenerateMonster.py to alter how to generate/sample from trained agents, Qlearner.py to alter the training process and get new agents, and MDP.py to alter the Markov Decision Process. 
6. Rerun Qlearner.py then GenerateMonster.py to see the impact of your changes.  
