"""
Module to plot the training data
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_BASE_FOLDER = "results"
# Custom reward function used in the trading environment
REWARD_FUNC = 2
RESULTS_FOLDER = f"{RESULTS_BASE_FOLDER}/reward_function_{REWARD_FUNC}"

TRAIN_DATA_FOLDER = f"train_data/reward_function_{REWARD_FUNC}"

def plot_train_data(datas, algorithm, tickers):
    """
    Plot the training data
    """
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    if not os.path.exists(f"{RESULTS_FOLDER}"):
        os.makedirs(f"{RESULTS_FOLDER}")

    plt.figure(figsize=(10, 6))
    for data in datas:
        plt.plot(data["Step"], data["Value"])
    plt.title(f"Learning Curve Comparison - Reward Function {REWARD_FUNC}")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend(tickers)
    plt.savefig(f"{RESULTS_FOLDER}/learning_curve.png")
    plt.close()

if __name__ == "__main__":
    TICKERS = ["COGN3", "ITUB4", "PETR4"]
    ALGORITHM = "PPO"
    DATAS = []
    for TICKER in TICKERS:
        DATA = pd.read_csv(f"{TRAIN_DATA_FOLDER}/{TICKER}.csv")
        DATAS.append(DATA)
    plot_train_data(datas=DATAS, algorithm=ALGORITHM, tickers=TICKERS)
