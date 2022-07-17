# HedgingPortfolioEnv
This repository was created for and used in the context of the bachelor thesis with the title "A Risk-Sensitive Approach for modeling the Hedging Portfolio Problem 
with Reinforcement Learning".

## OpenAI Gym Environment
Based on the `TradingEnv` and the `StocksEnv` from [gym-anytrading](https://github.com/AminHP/gym-anytrading), 
a new OpenAI Gym environment, called `HedgingPortfolioEnv`, was created.

- [HedgingPortfolioEnv.py](HedgingPortfolioEnv.py)
- [TradingEnv.py](TradingEnv.py)
- [StocksEnv.py](StocksEnv.py)

## DQN
For Reinforcement Learning training purposes, a simple implementation of the `DQN algorithm` from [minimalRL](https://github.com/seungeunrho/minimalRL/blob/master/dqn.py) was used.

To implement risk-sensitivity into the decision taking of the RL agents, the DQN algorithm was adapted, applying a new update rule,
as presented in the paper ["Model-Free Risk-Sensitive Reinforcement Learning"](https://deepmindsafetyresearch.medium.com/model-free-risk-sensitive-reinforcement-learning-5a12ba5ce662)

The simple DQN algorithm, the risk-sensitive DQN algorithm and training methods can be found in [dqn.py](dqn.py)
