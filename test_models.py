"""
Module for rendering the trading environment.
"""

import os
import shutil
import pandas as pd
import numpy as np
from gym_trading_env.renderer import Renderer
from stable_baselines3 import PPO
import gymnasium as gym
import matplotlib.pyplot as plt


TICKERS = ["COGN3", "ITUB4", "PETR4"]

RISK_FREE_RATE = 10.40 / 100 # CDI rate (annualized)

INITIAL_CAPITAL = 10_000

DATA_FOLDER = "data"
MODEL_BASE_FOLDER = "models"
RESULTS_BASE_FOLDER = "results"

START_DATE = "2014-05-30"
END_DATE = "2024-05-26"

ALGORITHM = "PPO"

# Custom reward function used in the trading environment
REWARD_FUNC = 2

MODEL_FOLDER = f"{MODEL_BASE_FOLDER}/reward_function_{REWARD_FUNC}"
RESULTS_FOLDER = f"{RESULTS_BASE_FOLDER}/reward_function_{REWARD_FUNC}"

RENDER_LOGS_DIR = f"render_logs/reward_function_{REWARD_FUNC}"

def custom_reward_function_01(history):
    """
    Custom reward function for the trading environment
    :param history: history object: History object of the trading environment
    :return: float: Reward value

    The reward function follows the formula:
    reward = (portfolio_valuation[-1] / portfolio_valuation[0])^3 / risk_free_rate

    # Full history documentation: https://gym-trading-env.readthedocs.io/en/latest/history.html
    """
    # Reward func 01
    return (history["portfolio_valuation", -1] / history["portfolio_valuation", 0])**3 / (RISK_FREE_RATE * 100)

def custom_reward_function_02(history):
    """
    Custom reward function for the trading environment
    :param history: history object: History object of the trading environment
    :return: float: Reward value

    The reward function follows the formula:
    reward = (portfolio_valuation[-1] / portfolio_valuation[0]) - (risk_free_rate / 365)

    # Full history documentation: https://gym-trading-env.readthedocs.io/en/latest/history.html
    """
    # Reward func 02
    return (history["portfolio_valuation", -1] / history["portfolio_valuation", 0]) - (RISK_FREE_RATE / 365)

def main(ticker):
    data = pd.read_feather(f"{DATA_FOLDER}/{ticker}_{START_DATE}_{END_DATE}.feather")
    data_size = data.shape[0]
    # Split the data into training and testing
    train_size = int(data_size * 0.75)
    test_data = data.iloc[train_size:]

    custom_reward_function = custom_reward_function_01 if REWARD_FUNC == 1 else custom_reward_function_02

    env = gym.make("TradingEnv",
        name= f"AlgoTradingTestingEnv{ticker}",
        df = test_data, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.03/100, # 0.03% per timestep (one timestep = 1 day here)
        portfolio_initial_value = INITIAL_CAPITAL, # The starting balance
        reward_function = custom_reward_function, # You can define your own reward function -> https://gym-trading-env.readthedocs.io/en/latest/customization.html#custom-reward-function
        windows = 14, # The amount of previous data to include in the observation
        verbose = 0, # Show the logs
    )

    env.unwrapped.add_metric("Position Changes", lambda history : np.sum(np.diff(history["position"]) != 0) )

    model = PPO.load(f"{MODEL_FOLDER}/{ticker}_{ALGORITHM}")

    portfolio_valuation = []
    done, truncated = False, False
    observation, info = env.reset()
    while not done and not truncated:
        # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
        position_index = model.predict(observation)[0] # At every timestep, pick a position index from your position list (=[-1, 0, 1]) using your trained model
        observation, reward, done, truncated, info = env.step(position_index)
        portfolio_valuation.append(info["portfolio_valuation"])

    # Render the trading environment
    env.unwrapped.save_for_render(dir=RENDER_LOGS_DIR)
    
    return portfolio_valuation

def calculate_interest_rate(initial_value, interest_rate, size):
    """
    Function to create a time series of money value with a fixed interest rate
    """
    return initial_value * (1 + interest_rate)**np.arange(size)

if __name__ == "__main__":

    if os.path.exists(RENDER_LOGS_DIR):
        shutil.rmtree(RENDER_LOGS_DIR)

    for TICKER in TICKERS:
        portfolio_valuation_over_time = main(TICKER)
        money_on_risk_free_rate = calculate_interest_rate(INITIAL_CAPITAL, RISK_FREE_RATE/365, len(portfolio_valuation_over_time))
        
        # Plot the valuation of the portfolio
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_valuation_over_time)
        plt.plot(money_on_risk_free_rate)
        plt.title(f"Portfolio Valuation Comparison - {TICKER} - Reward Function {REWARD_FUNC}")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Valuation (R$)")
        plt.grid()
        plt.legend([TICKER, "CDI"])
        plt.savefig(f"{RESULTS_FOLDER}/{TICKER}_portfolio_valuation.png")
        plt.close()

    renderer = Renderer(render_logs_dir=RENDER_LOGS_DIR)
    # Add Custom Metrics (Annualized metrics)
    renderer.add_metric(
        name = "Annual Market Return",
        function = lambda df : f"{ ((df['close'].iloc[-1] / df['close'].iloc[0])**(pd.Timedelta(days=365)/(df.index.values[-1] - df.index.values[0]))-1)*100:0.2f}%"
    )
    renderer.add_metric(
        name = "Annual Portfolio Return",
        function = lambda df : f"{((df['portfolio_valuation'].iloc[-1] / df['portfolio_valuation'].iloc[0])**(pd.Timedelta(days=365)/(df.index.values[-1] - df.index.values[0]))-1)*100:0.2f}%"
    )

    renderer.run()
