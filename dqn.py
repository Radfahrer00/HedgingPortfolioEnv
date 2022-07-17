import collections
import math
import random
import yfinance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from HedgingPortfolioEnv import HedgingPortfolioEnv
from StocksEnv import StocksEnv

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 30000
batch_size = 32


class ReplayBuffer():
    """
    This class models the replay memory for the DQN algorithm.
    """
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    """
    This class models a Deep Neural Network for the DQN algorithm.
    """
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    """
    This function is responsible for the training of the agent with a DQN algorithm.
    """
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_out = torch.reshape(q_out, (160, 2))
        q_out = q_out[0:32, :]
        q_a = q_out[:, :1]
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def buy_and_sell_training():
    """
    This function trains an agent in the StocksEnv with a DQN algorithm.
    """
    ticker = yfinance.Ticker("AAPL")
    df = ticker.history(start="2021-06-18")
    env = StocksEnv(df=df, window_size=5, frame_bound=(5, 105))

    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    trained_rewards = []

    print_interval = 50
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(1000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                trained_rewards.append(info.get("total_profit"))
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0
    env.close()

    return trained_rewards


def buy_sell_hold_and_options_training():
    """
    This function trains an agent in the HedgingPortfolioEnv with a DQN algorithm.
    """
    ticker = yfinance.Ticker("AAPL")
    df = ticker.history(start="2021-06-18")
    puts = ticker.option_chain(ticker.options[4]).puts
    env = HedgingPortfolioEnv(df=df, window_size=5, frame_bound=(5, 105), puts=puts)

    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    trained_rewards = []

    print_interval = 50
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(100):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                trained_rewards.append(info.get("total_profit"))
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0
    env.close()

    return trained_rewards


def scaled_logistic_sigmoid(delta, beta):
    """
    Helper function to calculate the risk-sensitive term for the risk-sensitive DQN algorithm.
    """
    sigmoid = 1 / (1 + math.exp(-beta * delta))
    return sigmoid


def train_risk_sensitive(q, q_target, memory, optimizer, beta):
    """
    The training function for the DQN algorithm is adapted to incorporate a risk-sensitive term,
    which is influenced by the parameter beta.
    """
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_out = torch.reshape(q_out, (160, 2))
        q_out = q_out[0:32, :]
        q_a = q_out[:, :1]
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

        max_r = torch.max(r).item()
        max_q = torch.max(max_q_prime).item()

        delta = max_r + gamma * max_q
        risk_sensitive_parameter = 2 * scaled_logistic_sigmoid(delta, beta)

        target = r + gamma * max_q_prime * done_mask * risk_sensitive_parameter
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def risk_sensitive_training(beta):
    """
    This function trains an agent in the HedgingPortfolioEnv with a DQN algorithm adapted to risk-sensitivity.
    The risk-sensitivity can be controlled through the parameter beta.
    For beta, values between -0.8 to 0.8 have been tested, however, it might be possible that smaller or larger values
    might also work and deliver interesting results.
    """
    ticker = yfinance.Ticker("AAPL")
    df = ticker.history(start="2021-06-18")
    puts = ticker.option_chain(ticker.options[4]).puts
    env = HedgingPortfolioEnv(df=df, window_size=5, frame_bound=(5, 105), puts=puts)

    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    risk_rewards = []
    risk_options = []

    print_interval = 50
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(1000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                risk_rewards.append(info.get("total_profit"))
                risk_options.extend(env.return_options())
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer, beta)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0
    env.close()

    return risk_rewards, risk_options
