import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

'''
Enum for possible actions - Buy and Sell
'''


class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2
    Buy_Options_one = 3
    Buy_Options_two = 4
    Buy_Options_three = 5
    Buy_Options_four = 6
    Buy_Options_five = 7
    Buy_Options_six = 8
    Buy_Options_seven = 9
    Buy_Options_eight = 10
    Buy_Options_nine = 11
    Buy_Options_ten = 12
    Buy_Options_eleven = 13
    Buy_Options_twelve = 14


class Positions(Enum):
    Short = 0
    Long = 1
    Options = 2

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class OptionsTradingEnv(gym.Env):
    """
    This class represents an abstract gym AI environment, which serves as a basis for environments, where options are
    to be traded.
    Extends the basic gym.Env and is based on the TradingEnv.
    Credit: https://github.com/AminHP/gym-anytrading
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._option_strike_price = None
        self._first_rendering = None
        self.history = None
        self._options_bought = []

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._option_strike_price = 0
        self._first_rendering = True
        self.history = {}
        self._options_bought = []
        return self._get_observation()

    def step(self, action):
        self._done = False
        self._current_tick += 1  # advance one tick

        if self._current_tick == self._end_tick:  # if it's the last tick, done = True
            self._done = True

        step_reward = self._calculate_reward(action)  # calculate the reward for the step with the action
        self._total_reward += step_reward  # add reward to total reward

        self._update_profit(action)  # calculate the profit (in percent)

        # If current position equals Short and action equals Buy, then a trade can happen (or vice versa)
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        # If the trade occurs, update current position and advance last trade tick to current tick
        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        # Buy options, if in a Long Position
        buy_options = False
        if 3 <= action <= 14 and self._position == Positions.Long and self._option_strike_price == 0:
            buy_options = True

        if buy_options:
            self._calculate_options_price(action)
            self._option_strike_price = self._get_options_strike_price(action)
            self._options_bought.append(self._option_strike_price)

        self._position_history.append(self._position)
        observation = self._get_observation()  # get observation of current state
        # Put reward and profit information into history
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info  # return the new observation, reward, if done and the info

    '''
    Returns extracted features from current_tick - window_size to current tick. Used to create Gym observations
    '''

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

    '''
    Create history if not already given, otherwise update values
    '''

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def print_options(self):
        print(self._options_bought)

    def return_options(self):
        return self._options_bought

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def _calculate_options_price(self, action):
        raise NotImplementedError

    def _get_options_strike_price(self, action):
        raise NotImplementedError
