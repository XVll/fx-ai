# Dashboard Improvement

This is the change list I have, I want you to implement these changes to the dashboard, in order to do that you need to implement each of them to required parts of the app.

Use clean architecture, I dont want to track each metric where in the code base, or where it is updated. Use data structures to define them so I will know what is there to change or add, remove.

Do not apply temporary patches if it is required apply clean refactoring to make it more readible, extensible.

Extensible part is important I should be able to add easily new features, stats, metrics, calculations.

**Price Precision:** Noted that all financial figures will be `%.2f`.

### **I. Header (Single Line)**

- **Left Side:**
    - `Model: [Current Model Name/ID]` (e.g., "PPO_Transformer_v3.2_ETH")
- **Right Side:**
    - `Session Time: [HH:MM:SS (Elapsed)]`

---

### **II. Body (Main Content Area - Organized into Panels)**

**Column Group 1:** 

- **Panel 1.1: Market Data**
    - **Title:** `📊 Market: [Symbol] - [Market Session (Pre/Regular/Post)]`
    - **Content:**
        - `Time (NY): [HH:MM:SS]`
        - `Price: $[Current Price.2f]` (e.g., `Arrow Up/Down` icon for tick change)
        - `Bid: $[Bid Price.2f]`
        - `Ask: $[Ask Price.2f]`
        - `Spread: $[Spread Value.2f]`
- **Panel 1.2: Current Position ([Symbol])**
    - **Title:** `💼 Position: [Symbol]`
    - **Content:**
        - `Side: [Side (Long/Short/Flat)]` (Color-coded)
        - `Quantity: [Quantity.4f]` (More precision for quantity if needed)
        - `Avg Entry Price: $[Avg Price.2f]`
        - `P&L vs Entry: $[P&L $.2f] ([P&L %.2f]%)` (Color-coded)
- **Panel 1.3: Portfolio**
    - **Title:** `📈 Portfolio`
    - **Content:**
        - `Total Equity: $[Portfolio Total Equity.2f]`
        - `Cash Balance: $[Cash Balance.2f]`
        - `Session P&L: $[P&L $.2f] ([P&L %.2f]%)` (Color-coded)
        - `Realized P&L : $[Realized P&L $.2f]`
        - `Unrealized P&L : $[Unrealized P&L $.2f]` (Color-coded)
        - `Sharpe Ratio : [Sharpe Ratio.2f]`
        - `Max Drawdown : [Max Drawdown %.2f]%`
        - `Trades : [Number of Trades]`

**Column Group 2:** 

- **Panel 2.1: Recent Actions**
    - **Title:** `⚡ Recent Actions`
    - **Recent Actions (Last 5):**
        - `Table: Step | Action Type | Size/Signal | Step Reward`
        - *(Example: `12345 | BUY | 75% | +0.15`)*
- **Panel 2.2:**
    - **Title:** `⚡ Recent Trades` Trades
    - **Recent Trades (Last 5):**
        - `Table: Time | Side | Qty | Symbol | Entry Price | Exit Price | P&L ($)`
        - *(Example: `10:08:30 | SELL | 10 | AAPL | 170.50 | 171.20 | +7.00`)*
- **Panel 2.3: Current/Last Episode Status**
    - **Title:** `🎬 Episode Analysis (Ep: [Current/Last Episode #])`
    - **Content:**
        - `Current Step: [Current Episode Step]`
        - `Cumulative Reward (Ep): [Current Episode Reward.2f]`
        - `Last Step Reward: [Last Step Reward.3f]`
        - **Panel 5.1: Episodes History**
            - **Title:** `📚 Episode History (Last 3)`
            - **Content:**
                - `Table: Ep # | Status | Reason | Ep Reward`
- **Panel 3.1: Training Progress**
    - **Title:** `⚙️ Training Progress`
    - **Content:**
        - `Mode: [Training / Evaluation / Idle]`
        - `Current Stage: [Training Stage Text]`
        - **Overall Progress:** Just like current one
        - **Current Stage Progress: Just like current one**
        - **Stage Status:** `[Dynamic text, e.g., "Rollout: 1500/2048 steps", "PPO Epoch: 3/10, Batch: 15/32"]`
        - `Updates: [Update Count]`
        - `Episode Counter`
        - `Global Step Counter`
- **Panel 3.2: PPO Core Metrics**
    - **Title:** `🧠 PPO Core Metrics (Current Batch/Update)`
    - **Content ( Historical Sparkline (if console allows)):**
        - *(For plots in console: show current value,  textual sparkline like `▂▃▅▇` if possible.)*
        - `Learning Rate: [LR.1e]`
        - `Mean Reward (Batch): [Mean Reward.2f] 
        - `Policy Loss: [Value.3f] | [Trend] 
        - `Value Loss: [Value.3f] | [Trend] 
        - `Total Loss: [Value.3f] | [Trend]
        - `Entropy: [Value.3f] | [Trend] 
        - `Clip Fraction: [Value.3f] | [Trend] 
        - `Approx KL: [Value.3f] | [Trend] 
        - `Value Explained Var.: [Value.3f] | [Trend] 

- **Panel 3.3 Reward Component Breakdown (Table or List):**
    - **Title:** `🏆 Rward System System`
    - `COMPONENT_NAME | TYPE | TOTAL_IMPACT (Ep) | % OF TOTAL_REW | AVG_MAG (when active) | TIMES_TRIGGERED`
    - *(Example: `penalty_holding_inaction | Penalty | -1.50 | -9.8% | -0.01 | 150`)*
- **Panel 4.1: Reward System Analysis**
    - **Title:** `🏆 Action Analysis`
    - **Content:**
        - **Action Bias Summary :**
        - `Invalid Actions (Count): [Number of Invalid Actions Attempted]`
            
            `Action Bias:
              ACTION | COUNT | % STEPS | MEAN_REW | TOTAL_REW | POS_REW_RATE (%)
              -------------------------------------------------------------------
              HOLD   |  150  |  60.0   |   0.01   |    1.50   |   60.0
              BUY    |   50  |  20.0   |   0.25   |   12.50   |   80.0
              SELL   |   50  |  20.0   |   0.03   |    1.50   |   55.0`
            

---

### **III. Footer (Single Line)**

- `Steps/Sec: [Steps per Second.1f]`
- `Time/Update: [Avg Time per Update.2f]s`
- `Time/Episode (Avg): [Avg Time per Episode.1f]s`