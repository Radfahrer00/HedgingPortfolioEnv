import numpy as np

from OptionsTradingEnv import OptionsTradingEnv, Actions, Positions


class HedgingPortfolioEnv(OptionsTradingEnv):
    """
    This class represents a gym AI environment, where stocks can be bought, sold, held and put options bought.
    Extends the OptionsTradingEnv and is based on the StocksEnv.
    Credit: https://github.com/AminHP/gym-anytrading
    """

    def __init__(self, df, window_size, frame_bound, puts):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        # Trading fees
        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

        self.puts = puts

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()  # Converts the Close column prices to numpy array

        prices[self.frame_bound[0] - self.window_size]
        prices = prices[
                 self.frame_bound[0] - self.window_size:self.frame_bound[1]]  # reduces prices to length of frame bound

        diff = np.insert(np.diff(prices), 0, 0)  # calculates difference from current price to price of the day before
        signal_features = np.column_stack((prices, diff))  # Joins the price with the price difference in a 2D-array

        return prices, signal_features

    def _calculate_reward(self, action):
        step_reward = 0

        # Make a trade if the action is opposite to position
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        # Calculate reward if the trade happens
        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price  # Calculate current price difference to price of yesterday

            # Only add reward, if shares where sold
            if self._position == Positions.Long:
                # Check if options strike price is higher than current price (means that options are held)
                if current_price <= self._option_strike_price:
                    price_diff = self._option_strike_price - last_trade_price
                    self._option_strike_price = 0

                step_reward += price_diff

        return step_reward

    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            # Calculate total shares based on the number of shares that can be bought
            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

    def _calculate_options_price(self, action):
        if action == 3:
            price = self.puts['lastPrice'].iloc[2]
        elif action == 4:
            price = self.puts['lastPrice'].iloc[3]
        elif action == 5:
            price = self.puts['lastPrice'].iloc[4]
        elif action == 6:
            price = self.puts['lastPrice'].iloc[5]
        elif action == 7:
            price = self.puts['lastPrice'].iloc[6]
        elif action == 8:
            price = self.puts['lastPrice'].iloc[7]
        elif action == 9:
            price = self.puts['lastPrice'].iloc[8]
        elif action == 10:
            price = self.puts['lastPrice'].iloc[9]
        elif action == 11:
            price = self.puts['lastPrice'].iloc[10]
        elif action == 12:
            price = self.puts['lastPrice'].iloc[11]
        elif action == 13:
            price = self.puts['lastPrice'].iloc[12]
        elif action == 14:
            price = self.puts['lastPrice'].iloc[13]
        else:
            price = 0

        self._total_reward -= price / 10

        last_trade_price = self.prices[self._last_trade_tick]
        shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
        number_of_options = int(shares * 1000)
        self._total_profit -= price / (self._total_profit * 1000)

    def _get_options_strike_price(self, action):
        if action == 3:
            strike_price = self.puts['strike'].iloc[2]
        elif action == 4:
            strike_price = self.puts['strike'].iloc[3]
        elif action == 5:
            strike_price = self.puts['strike'].iloc[4]
        elif action == 6:
            strike_price = self.puts['strike'].iloc[5]
        elif action == 7:
            strike_price = self.puts['strike'].iloc[6]
        elif action == 8:
            strike_price = self.puts['strike'].iloc[7]
        elif action == 9:
            strike_price = self.puts['strike'].iloc[8]
        elif action == 10:
            strike_price = self.puts['strike'].iloc[9]
        elif action == 11:
            strike_price = self.puts['strike'].iloc[10]
        elif action == 12:
            strike_price = self.puts['strike'].iloc[11]
        elif action == 13:
            strike_price = self.puts['strike'].iloc[12]
        elif action == 14:
            strike_price = self.puts['strike'].iloc[13]
        else:
            strike_price = 0

        return strike_price
