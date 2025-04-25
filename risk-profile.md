# RL Agent Risk Profile and Trading Strategy

This document outlines the risk profile and inherent trading strategy characteristics of the RL agent as defined primarily within the `rl_agent/environment/trading_env.py` and configured via `train_config.json` or command-line arguments.

## Core Trading Strategy

The agent employs a strategy learned through reinforcement learning (specifically algorithms like PPO, DQN, SAC, etc., using Stable Baselines3). Its goal is to maximize a cumulative reward signal over time.

-   **Decision Basis:** Decisions (Buy, Sell, Hold) are based on observing a sequence (`sequence_length`) of market features (e.g., OHLCV, technical indicators). The agent learns patterns and correlations within these features to predict profitable actions.
-   **Action Space:**
    -   `Buy`: Enter a long position using a fixed fraction (`max_position`) of the current cash balance.
    -   `Sell`: Exit the *entire* current long position.
    -   `Hold`: Maintain the current position (either cash or holding shares).
-   **Observation:** The agent directly observes market features but *not* its current portfolio value, balance, or shares held. This information is used internally for reward calculation and portfolio updates but doesn't directly influence the policy network's input at each step.

## Key Risk Management Mechanisms

Risk is primarily managed through a combination of explicit parameters and the structure of the reward function:

1.  **Position Sizing (`max_position`):**
    -   This is a critical risk control. It limits the capital allocated to any single 'Buy' action to a fraction of the available balance (e.g., default might be 1.0, meaning full balance, or a smaller fraction like 0.5).
    -   It prevents catastrophic loss from a single bad trade by limiting exposure.

2.  **Transaction Costs (`transaction_fee`):**
    -   A realistic transaction fee (e.g., 0.00075 or 0.075% for Binance Spot) is applied to both buy and sell actions.
    -   This implicitly discourages overly frequent trading (scalping) which can erode profits and might indicate noisy/unreliable signals.
    -   The `fee_penalty_weight` in the reward function explicitly penalizes these costs.

3.  **Reward Function Shaping:** The reward function is designed to balance profit-seeking with risk aversion:
    -   **`drawdown_penalty_weight`:** Directly penalizes the agent if its current portfolio value drops below the historical maximum portfolio value achieved during the episode. This encourages capital preservation and discourages significant drawdowns. A higher weight increases risk aversion.
    -   **`sharpe_reward_weight`:** Rewards the agent based on the Sharpe ratio calculated over a rolling window (`sharpe_window`). This promotes achieving returns that are high relative to the volatility (risk) taken, encouraging risk-adjusted performance.
    -   **`consistency_penalty_weight`:** Penalizes the agent for rapidly flipping between buy and sell actions within a short period (`consistency_threshold`). This discourages erratic behavior and encourages holding positions for a reasonable duration, potentially filtering out noise.
    -   **`trade_penalty_weight`:** Applies a small, fixed penalty for *any* trade (buy or sell). This can further discourage excessive trading if `transaction_fee` alone isn't sufficient.

## Risk-Seeking Elements (and Controls)

While managing risk, the agent must also take positions to generate profit:

-   **Profit Focus (`portfolio_change_weight`, `profit_bonus_weight`):** These reward components directly encourage increasing portfolio value and realizing profits on trades. They are the primary drivers for entering positions.
-   **Benchmark Comparison (`benchmark_reward_weight`):** Encourages the agent to outperform a simple buy-and-hold strategy, potentially leading it to take trades even in sideways markets if it anticipates short-term gains.
-   **Exploration (`exploration_bonus_weight`, RL algorithm):** During training, exploration mechanisms (like epsilon-greedy for DQN or stochasticity in PPO/SAC policies, plus the explicit `exploration_bonus_weight`) encourage the agent to try different actions, including potentially suboptimal or risky ones, to discover better strategies. This exploration naturally decreases as training progresses.

## Configurability

The overall risk profile is **highly configurable**. The default values in `train_config.json` define a baseline, but altering parameters via command-line arguments significantly changes the agent's behavior:
-   Increasing `max_position` increases potential profit/loss per trade.
-   Increasing `drawdown_penalty_weight` or `sharpe_reward_weight` makes the agent more conservative.
-   Increasing `portfolio_change_weight` makes the agent more aggressive in seeking portfolio growth.
-   Adjusting `consistency_threshold` and `idle_threshold` influences trading frequency.

## Limitations & Assumptions

-   **Slippage:** The model only accounts for fixed `transaction_fee`. It does not model market impact or variable slippage based on trade size or market volatility.
-   **Liquidity:** Assumes trades can always be executed at the recorded 'close' price for the interval.
-   **Catastrophic Events:** The model learns from the provided data and may not be robust to "black swan" events or market conditions entirely dissimilar to its training data.
-   **No Shorting:** The current environment only permits long positions (Buy, Sell, Hold).

## Summary

The agent's strategy is to identify profitable trading opportunities based on learned patterns in market features. Risk is managed through fixed position sizing, transaction cost awareness, and a reward function explicitly penalizing drawdowns, volatility (via Sharpe), and excessive/erratic trading, while still rewarding profit generation. The balance between risk aversion and profit-seeking is heavily influenced by the chosen configuration parameters. 