import statistics

import yfinance
import numpy as np
import pandas as pd

import dqn
from HedgingPortfolioEnv import HedgingPortfolioEnv
from StocksEnv import StocksEnv


def random_buy_and_sell_agent(episodes):
    """
    This function creates an environment to trade the Apple ("AAPL") stock for 100 days, starting on the 25th
    of June, 2021. In this environment, the agent can only buy and sell the shares and these actions are
    performed randomly.
    :param episodes: Number of times, the agent should randomly trade the stock for 100 days
    :return: the profits achieved in each episode
    """
    ticker = yfinance.Ticker("AAPL")
    df = ticker.history(start="2021-06-25")

    env = StocksEnv(df=df, window_size=0, frame_bound=(0, 100))

    episode_profits = []
    for run in range(episodes):
        state = env.reset()
        while True:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            if done:
                episode_profits.append(info.get("total_profit"))
                break

    episode_profits = np.array(episode_profits)
    return episode_profits


def random_buy_sell_hold_and_options(episodes):
    """
    This function creates an environment to trade the Apple ("AAPL") stock for 100 days, starting on the 25th
    of June, 2021. In this environment, the agent can buy, sell and hold the shares and buy put options.
    These actions are performed randomly.
    :param episodes: Number of times, the agent should randomly trade the stock for 100 days
    :return: the profits achieved in each episode
    """
    ticker = yfinance.Ticker("AAPL")
    df = ticker.history(start="2021-06-25")
    puts = ticker.option_chain(ticker.options[4]).puts

    env = HedgingPortfolioEnv(df=df, window_size=0, frame_bound=(0, 100), puts=puts)

    episode_profits = []
    for run in range(episodes):
        state = env.reset()
        while True:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            if done:
                episode_profits.append(info.get("total_profit"))
                break

    episode_profits = np.array(episode_profits)
    return episode_profits


def print_info(profits):
    """
    This function returns all important key figures for the analysis of the profits
    """
    print("MAX Profit:")
    print(max(profits))
    print("MIN Profit:")
    print(min(profits))
    print("Average Profit:")
    print(np.average(profits))
    print("Median Profit:")
    print(np.median(profits))
    print("Profit Range:")
    print(max(profits) - min(profits))

    # Calculating the standard deviation
    standard_dev = statistics.stdev(profits)
    print("Standard Deviation: " + str(standard_dev))

    # Calculating the Sharpe Ratio
    risk_free = 0.00045
    sharpe = []
    for profit in profits:
        return_ = profit - 1
        sharpe_ratio = (return_ - risk_free) / standard_dev
        sharpe.append(sharpe_ratio)
    print("Sharpe Ratio: " + str(np.average(sharpe)))
    print("")

    # Calculating the VaR
    profits_df = pd.DataFrame()
    profits_df['profits'] = profits
    profits_df.sort_values('profits', inplace=True, ascending=True)
    var90 = profits_df['profits'].quantile(0.1)
    var95 = profits_df['profits'].quantile(0.05)
    var99 = profits_df['profits'].quantile(0.01)
    print('Confidence Level: 90%        Value at Risk: ' + str(var90))
    print('Confidence Level: 95%        Value at Risk: ' + str(var95))
    print('Confidence Level: 99%        Value at Risk: ' + str(var99))


if __name__ == '__main__':
    random_buy_and_sell_profits = random_buy_and_sell_agent(10)
    print_info(random_buy_and_sell_profits)

    """
    random_buy_sell_hold_and_options_profits = random_buy_sell_hold_and_options(10)
    trained_buy_and_sell_profits = dqn.buy_and_sell_training()
    trained_buy_sell_hold_and_options_profits = dqn.buy_sell_hold_and_options_training()
    trained_risk_sensitive_profits = dqn.risk_sensitive_training(-0.8)
    """
