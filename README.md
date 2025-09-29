# RL-Algotrading: Aprendizado por Reforço para Trading Quantitativo

Este projeto utiliza técnicas de Aprendizado por Reforço (Reinforcement Learning) para desenvolver, treinar e avaliar agentes de trading quantitativo no mercado financeiro brasileiro. O objetivo é criar estratégias automatizadas que aprendam a maximizar o retorno de portfólios utilizando dados históricos de ações.

## Principais Funcionalidades
- Coleta e pré-processamento de dados históricos de ações (B3).
- Implementação de múltiplas funções de recompensa para avaliação de estratégias.
- Treinamento de agentes com algoritmos de RL (ex: PPO).
- Visualização de curvas de aprendizado e avaliação de desempenho.
- Armazenamento de logs, modelos e resultados para análise posterior.

## Estrutura do Projeto
```
├── data_collector.py           # Script para coleta de dados
├── main.py                     # Execução principal do treinamento e avaliação
├── train_plotter_all_together.py # Visualização dos resultados
├── test_models.py              # Testes dos modelos treinados
├── models/                     # Modelos treinados organizados por função de recompensa
├── data/                       # Dados históricos em formato feather
├── train_data/                 # Dados de treino por função de recompensa
├── results/                    # Resultados e gráficos gerados
├── render_logs/                # Logs de execução dos agentes
├── tensorboard_log/            # Logs para visualização no TensorBoard
├── Reinforcement Learning for Quantitative Trading.ipynb # Notebook explicativo
```

## Como Executar
1. Instale as dependências necessárias (exemplo: `pip install -r requirements.txt`).
2. Execute `data_collector.py` para baixar/preparar os dados.
3. Execute `main.py` para treinar os agentes.
4. Utilize `train_plotter_all_together.py` para visualizar os resultados.
5. Analise os logs e gráficos em `results/` e `tensorboard_log/`.

## Tecnologias Utilizadas
- Python 3.x
- Stable Baselines3 (RL)
- Pandas, NumPy
- Matplotlib, Seaborn
- TensorBoard

## Sobre o Autor
Projeto desenvolvido para portfólio profissional. Sinta-se à vontade para entrar em contato para colaborações ou dúvidas!

---

> **Nota:** Este projeto é para fins educacionais e de pesquisa. Não constitui recomendação de investimento.

---

# RL-Algotrading: Reinforcement Learning for Quantitative Trading

This project leverages Reinforcement Learning techniques to develop, train, and evaluate quantitative trading agents in the Brazilian stock market. The goal is to create automated strategies that learn to maximize portfolio returns using historical stock data.

## Main Features
- Collection and preprocessing of historical stock data (B3).
- Implementation of multiple reward functions for strategy evaluation.
- Training of agents with RL algorithms (e.g., PPO).
- Visualization of learning curves and performance evaluation.
- Storage of logs, models, and results for further analysis.

## Project Structure
```
├── data_collector.py           # Data collection script
├── main.py                     # Main training and evaluation execution
├── train_plotter_all_together.py # Results visualization
├── test_models.py              # Testing of trained models
├── models/                     # Trained models organized by reward function
├── data/                       # Historical data in feather format
├── train_data/                 # Training data by reward function
├── results/                    # Generated results and charts
├── render_logs/                # Agent execution logs
├── tensorboard_log/            # Logs for TensorBoard visualization
├── Reinforcement Learning for Quantitative Trading.ipynb # Explanatory notebook
```

## How to Run
1. Install the required dependencies (e.g., `pip install -r requirements.txt`).
2. Run `data_collector.py` to download/prepare the data.
3. Run `main.py` to train the agents.
4. Use `train_plotter_all_together.py` to visualize the results.
5. Analyze the logs and charts in `results/` and `tensorboard_log/`.

## Technologies Used
- Python 3.x
- Stable Baselines3 (RL)
- Pandas, NumPy
- Matplotlib, Seaborn
- TensorBoard

## About the Author
Project developed for professional portfolio purposes. Feel free to get in touch for collaborations or questions!

---

> **Note:** This project is for educational and research purposes only. It does not constitute investment advice.