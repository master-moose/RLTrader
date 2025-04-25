# RL Agent Risk Profile and Trading Strategy

This document outlines the risk profile and inherent trading strategy characteristics of the RL agent as defined primarily within the `rl_agent/environment/trading_env.py` and configured via `train_config.json` or command-line arguments. **This version reflects the environment supporting bidirectional trading (Long and Short positions).**

## Core Trading Strategy

The agent employs a strategy learned through reinforcement learning (specifically algorithms like PPO, DQN, SAC, etc., using Stable Baselines3). Its goal is to maximize a cumulative reward signal over time by entering and exiting both long and short positions.

-   **Decision Basis:** Decisions (Hold, Go Long, Go Short, Close Position) are based on observing a sequence (`sequence_length`) of market features (e.g., OHLCV, technical indicators) *plus* the agent's current position state (Flat, Long, or Short) and normalized entry price.
-   **Action Space:**
    -   `Hold`: Maintain the current position (Long, Short, or Flat).
    -   `Go Long`: Enter a long position using a fraction (`max_position`) of the current *cash balance*, only if currently Flat.
    -   `Go Short`: Enter a short position, typically sized based on a fraction (`max_position`) of the current *cash balance* (simulating margin implicitly), only if currently Flat.
    -   `Close Position`: Exit the current Long or Short position to return to Flat.
-   **Observation:** The agent observes market features, its position type (-1, 0, 1), and a normalized representation of its entry price relative to the current price. It does *not* directly observe its absolute balance or portfolio value in the standard configuration.

## Key Risk Management Mechanisms

Risk is managed through explicit parameters, the environment structure, and the reward function:

1.  **Position Sizing (`max_position`):**
    -   Limits the capital (if going long) or the *value* of the assets borrowed (if going short) relative to the current balance when *entering* a position.
    -   This helps prevent excessive leverage or exposure on any single trade entry.

2.  **Transaction Costs (`transaction_fee`):**
    -   A realistic fee is applied when *entering* a long/short position and when *closing* a position.
    -   Discourages hyperactive trading. The `fee_penalty_weight` in the reward explicitly penalizes these costs relative to portfolio value.

3.  **Reward Function Shaping:** Balances profit-seeking with risk aversion:
    -   **`drawdown_penalty_weight`:** Penalizes drops from the peak portfolio value (calculated considering both long and short PnL). Encourages capital preservation.
    -   **`sharpe_reward_weight`:** Rewards higher risk-adjusted returns based on portfolio value volatility. Promotes smoother equity curves.
    -   **`trade_penalty_weight`:** Applies a fixed penalty for *entering* a long or short trade. Can further discourage excessive entries.
    -   **`idle_penalty_weight`:** Penalizes staying *Flat* (holding only cash) for too many consecutive steps (`idle_threshold`). Encourages market participation.

4.  **Short Position Mechanics:**
    -   **Implicit Margin:** The environment allows shorting based on `max_position` * balance, but doesn't explicitly model margin calls or liquidation based on balance depletion *while holding the short*. Risk is primarily managed by the PnL calculation affecting portfolio value and triggering the drawdown penalty/termination.
    -   **Buy-to-Cover Check:** The agent *cannot* close a short position if its current balance is insufficient to buy back the shares at the current market price plus fees. This prevents the balance from going negative due to short losses exceeding available cash but can lead to the agent getting "stuck" in a losing short if not managed by other reward components or termination conditions.

5.  **Episode Termination on Drawdown:** The episode automatically terminates if the portfolio drawdown exceeds a defined threshold (e.g., 50%), limiting potential losses within a single run.

## Risk-Seeking Elements (and Controls)

-   **Profit Focus (`portfolio_change_weight`, `profit_bonus_weight`):** These components reward increasing portfolio value and realizing profits on *closed* trades (both long and short).
-   **Exploration:** RL algorithms naturally explore during training, which can involve taking risks.

## Configurability

The risk profile remains highly configurable through `train_config.json` and command-line arguments, adjusting the weights of different reward components and the `max_position` size.

## Limitations & Assumptions

-   **Slippage:** Still only models fixed `transaction_fee`.
-   **Liquidity:** Assumes trades execute at the interval's 'close' price.
-   **Margin/Liquidation:** Shorting mechanics are simplified; no explicit margin calls or forced liquidations beyond the insufficient balance check when closing.
-   **Catastrophic Events:** Still vulnerable to events outside the training data distribution.
-   **Single Asset:** Only trades one asset pair.

## Summary

The agent learns a bidirectional strategy to identify profitable long and short opportunities based on market features and its current position. Risk is managed via position sizing at entry, transaction costs, reward penalties for drawdowns and excessive entry frequency, reward bonuses for risk-adjusted returns (Sharpe), and episode termination on high drawdown. The ability to short introduces new risk dimensions (e.g., theoretically unlimited loss, though capped by episode structure and balance checks), managed implicitly through portfolio value calculations and reward shaping. The specific risk appetite is tunable via configuration. 