
## Content Location

- My sorted content is located in the good and bad folders in the policy network for 100 and 125 points.
- I only sorted the content into good and bad for the runs where the agent was allowed up to 100 and up to 125 points.

## Modifications

- These were mentioned in the README, but I will mention them again here.
- I modified the MDP to introduce a constraint so that they is a maximum amount of points the creature is allowed to have.
- I also changed the hyperparameters used to train the model in the tabular case.
- Lastly, for the simulated battle, I also changed it so that during training, it had a chance to fight one of four different monsters instead of just fighting the balanced one only. My hope was that this would stop the algorithm from exploiting the specific values used by the balanced agent to generalize to more interesting content overall.
- I also implemented DQN to train the Q function instead of the tabular method used by the original implementation. As a result of this, I also modified how creatures were generated. I removed the max value threshold case, because the values of my function were not scaled in the same manner.
- Both my updated tabular and the Q Learning approaches use the MDP file in the qTabular folder.

## Training Information

- All monsters were generated using the policy network with either 100 or 125 points allowed, trained on 100 episodes.
- They were generated by fighting ONLY the balanced monster, even though in each battle phase they were trained against four potential opponents. 
- They were also only allowed 10 attempts, with a rollout length of 100 (so they could always allocate all or most of their points even if they randomly happened to roll all 0s as the starting stats).

## How Did I Classify My Content:

- The first I took into account was the final reward which was earned by each agent. This gave an initial good first idea.
- I also tried to account for how interesting of a build I thought the monster was. Some of them had very high final rewards, but I ranked them as bad content. This was typically because I thought they had a silly or uninteresting build. This included having very high armor, but something like only one attack, so they would basically just slowly wore down the opponent. If a player had to play against this, or play using this, this would be a very boring and not fun encounter, especially if they were playing against it and did not have a monster which could get through the defence if this generated one.
- Counter to this last point, some I ranked highly because I thought it was interesting that they went for a more tank like build, like high health and/or high armour (but not so over the top that it would likely not be fun).
- I also labeled content that I thought would generalize well to monsters other than the balanced monster. If I felt the content really only exploited some weakness in the stats of the balanced agent, I ranked it as bad content. This may not be very fair, as the MDP did not structure the problem such that it trained against a very broad base of content.
- Generally, I also labeled builds that were more balanced as good, as opposed to just running a glass cannon like build.
- The glass cannon style builds were usually the ones which ranked the highest in terms of reward, but I typically ranked them lower because I do not think it would be fun for a long amount of time to play against them or as them for very long.
- My focus was mainly on what I thought would create an interesting, but not too difficult creature to battle against if I was using the balanced monster.
- Generally, the content which was allowed more points (125 instead of 100) lead to more balanced, but less interesting builds overall.
- As a result, even though the final reward is estimated to be higher on average, I still ranked it lower because it created much less varied and more generic creatures.
- This makes sense as a result of the MDP though, so fundamentally a different MDP is needed to generate more interesting content.
