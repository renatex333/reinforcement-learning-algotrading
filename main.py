"""
Script to train the trading agent using the Reinforcement Learning algorithms
"""

import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime, timedelta
from data_collector import get_data, name


TICKERS = ["COGN3.SA", "ITUB4.SA", "PETR4.SA"]

DATA_FOLDER = "data"
MODEL_BASE_FOLDER = "models"

N_YEARS = 10

RISK_FREE_RATE = 10.40 / 100 # CDI rate (annualized)

INITIAL_CAPITAL = 10_000

# Custom reward function used in the trading environment
REWARD_FUNC = 1

MODEL_FOLDER = f"{MODEL_BASE_FOLDER}/reward_function_{REWARD_FUNC}"

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

def main(algorithm: str = "PPO"):
    """
    Main function to train and test the trading agent.
    """

    # start_date = (datetime.now() - timedelta(days=365*N_YEARS)).strftime("%Y-%m-%d")
    # end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    start_date = "2014-05-30"
    end_date = "2024-05-26"

    for ticker in TICKERS:
        # Retrieve historical data
        data = get_data(ticker, start_date, end_date, DATA_FOLDER)

        # Split the data into training and testing
        data_size = data.shape[0]
        train_size = int(data_size * 0.75)
        train_data = data.iloc[:train_size]

        # Train trading agent
        train(train_data, name(ticker), algorithm=algorithm)


def train(data: pd.DataFrame, ticker: str, algorithm: str):
    """
    Train the trading agent
    :param data: pd.DataFrame: Historical data of the stock
    :param ticker: str: Ticker of the stock
    :param algorithm: str: Reinforcement Learning algorithm to use
    """
    
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    file_path = f"{MODEL_FOLDER}/{ticker}_{algorithm}"

    train_size = data.shape[0]
    n_envs = 1
    env = create_env(data, ticker, n_envs=n_envs, verbose=0)
    model = None
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=train_size,
            batch_size= min(train_size*n_envs, 16384),
            n_epochs=20,
            gamma=0.9999,
            verbose=0,
            tensorboard_log="tensorboard_log",
        )
    elif algorithm == "RecurentPPO":
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=1e-4,
            n_steps=train_size,
            batch_size=min(train_size*n_envs, 16384),
            n_epochs=20,
            gamma=0.9999,
            verbose=0,
            tensorboard_log="tensorboard_log",
        )
    else:
        raise ValueError(f"Algorithm {algorithm} not supported.")

    timesteps = train_size*3_000
    print(f"Training on {ticker} using {algorithm} algorithm. Total timesteps: {timesteps}")
    model.learn(total_timesteps=timesteps)

    print(f"Saving model for {ticker} using {algorithm} algorithm with reward function {REWARD_FUNC}...")
    model.save(file_path)

def create_env(data: pd.DataFrame, ticker: str, n_envs: int = 1, verbose: int = 0):
    """
    Create the trading environment
    :param data: pd.DataFrame: Historical data of the stock
    :param ticker: str: Ticker of the stock
    :return: gym.Env: Trading environment
    """

    custom_reward_function = custom_reward_function_01 if REWARD_FUNC == 1 else custom_reward_function_02

    vec_env = make_vec_env(
        "TradingEnv",
        n_envs=n_envs,
        env_kwargs={
            "name": f"{ticker}-TrainingEnv",
            "df": data,
            "positions": [-1, 0, 1],
            "trading_fees": 0.01/100,
            "borrow_interest_rate": 0.03/100,
            "portfolio_initial_value": INITIAL_CAPITAL,
            "reward_function": custom_reward_function,
            "windows": 14,
            "verbose": verbose,
            "render_mode": "logs",
        }
    )

    vec_env_add_metric = vec_env.env_method("get_wrapper_attr", "add_metric")[0]
    vec_env_add_metric(
        "Position Changes",
        lambda history : np.sum(np.diff(history["position"]) != 0)
    )

    return vec_env

if __name__ == "__main__":
    for algo in ["PPO"]:
        main(algorithm=algo)
