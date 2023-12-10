## RL-branch

### Overview

RL-branch stands as a distributed reinforcement learning framework, which accomplishes the implementation of distributed reinforcement learning algorithms by amalgamating distributed data sampling and centralized training. 

1. Components:

- Functioning as the primary program entry point, the Runner is tasked with sampling from the environment and calling upon remote Agents to execute distributed sampling and centralized training. 

- The Agent embodies the intelligent algorithm, allowing users to engage in training with intelligent algorithms effortlessly, eliminating the need for coding.

2. Algorithm Support:

- The platform extends support to various algorithms such as Q-Learning, DQN, Rainbow, SAC, DDPG, TRPO, TD3, BC, GAIL and more, leveraging the intelligent algorithms furnished by the platform.
- Users can leverage these algorithms without the need for manual coding.

### Instruction for Operation

1. main function: Create multiple Runners and launch them automatically in a Ray cluster using the 'makeRunner' function.

       if __name__ == '__main__':
       	# Create two GymRunners using makeRunner
       	runners = makeRunner(GymRunner, num=2)
       	# run
       	run()

2. To Run:

- Install the package:

  ```
  pip install -e .
  ```

- navigate to the ddpg directory:

  ```
  cd example/ddpg
  ```

- start the ray cluster(head node) on port 5876:

  ```
  ray start --head --port=6379
  ```

- run the DDPG example script:

  ```
  python ddpg.py
  ```

  