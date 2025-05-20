  self.static_feature_names: List[str] = [
            "S_Time_Of_Day_Seconds_Encoded_Sin",
            "S_Time_Of_Day_Seconds_Encoded_Cos",
            "S_Market_Cap_Million",
        ]
        # "S_Initial_PreMarket_Gap_Pct",  # Todo: Removed for now, requires to fetch previous day's data.
        # "S_Regular_Open_Gap_Pct",
        self.hf_feature_names: List[str] = [
            "HF_1s_Price_Velocity", # Rate of price change over the last 1 second
            "HF_1s_Price_Acceleration", # Second derivative of price (change in velocity)
            "HF_1s_Volume_Velocity", # Rate of volume change over the last 1 second
            "HF_1s_Volume_Acceleration", # Second derivative of volume (change in velocity)
            "HF_1s_HighLow_Spread_Rel",
            "HF_Tape_1s_Trades_Count_Ratio_To_Own_Avg", 
            "HF_Tape_1s_Trades_Count_Delta_Pct", 
            "HF_Tape_1s_Normalized_Volume_Imbalance",
            "HF_Tape_1s_Normalized_Volume_Imbalance_Delta", 
            "HF_Tape_1s_Avg_Trade_Size_Ratio_To_Own_Avg", 
            "HF_Tape_1s_Avg_Trade_Size_Delta_Pct",
            "HF_Tape_1s_Large_Trade_Count", 
            "HF_Tape_1s_Large_Trade_Net_Volume_Ratio_To_Total_Vol", 
            "HF_Tape_1s_Trades_VWAP",
            "HF_Quote_1s_Spread_Rel", 
            "HF_Quote_1s_Spread_Rel_Delta", 
            "HF_Quote_1s_Quote_Imbalance_Value_Ratio",
            "HF_Quote_1s_Quote_Imbalance_Value_Ratio_Delta", 
            "HF_Quote_1s_Bid_Value_USD_Ratio_To_Own_Avg", 
            "HF_Quote_1s_Ask_Value_USD_Ratio_To_Own_Avg",
        ]
        self.mf_feature_names: List[str] = [
            "MF_1m_PriceChange_Pct", "MF_5m_PriceChange_Pct", "MF_1m_PriceChange_Pct_Delta", "MF_5m_PriceChange_Pct_Delta",
            "MF_1m_Position_In_CurrentCandle_Range", "MF_5m_Position_In_CurrentCandle_Range", "MF_1m_Position_In_PreviousCandle_Range",
            "MF_5m_Position_In_PreviousCandle_Range",
            "MF_1m_Dist_To_EMA9_Pct", "MF_1m_Dist_To_EMA20_Pct", "MF_5m_Dist_To_EMA9_Pct", "MF_5m_Dist_To_EMA20_Pct",
            "MF_Dist_To_Rolling_HF_High_Pct", "MF_Dist_To_Rolling_HF_Low_Pct",
            "MF_1m_MACD_Line", "MF_1m_MACD_Signal", "MF_1m_MACD_Hist", "MF_1m_ATR_Pct", "MF_5m_ATR_Pct",
            "MF_1m_BodySize_Rel", "MF_1m_UpperWick_Rel", "MF_1m_LowerWick_Rel", "MF_5m_BodySize_Rel", "MF_5m_UpperWick_Rel", "MF_5m_LowerWick_Rel",
            "MF_1m_BarVol_Ratio_To_TodaySoFarVol",  # RENAMED and logic changed
            "MF_5m_BarVol_Ratio_To_TodaySoFarVol",  # RENAMED and logic changed
            "MF_1m_Volume_Rel_To_Avg_Recent_Bars", "MF_5m_Volume_Rel_To_Avg_Recent_Bars",
            "MF_1m_Dist_To_Recent_SwingHigh_Pct", "MF_1m_Dist_To_Recent_SwingLow_Pct", "MF_5m_Dist_To_Recent_SwingHigh_Pct",
            "MF_5m_Dist_To_Recent_SwingLow_Pct",
        ]
        self.lf_feature_names: List[str] = [
            "LF_Position_In_Daily_Range", "LF_Position_In_PrevDay_Range", "LF_Pct_Change_From_Prev_Close",
            "LF_RVol_Pct_From_Avg_10d_Timed", "LF_Dist_To_Session_VWAP_Pct",
            "LF_Daily_Dist_To_EMA9_Pct", "LF_Daily_Dist_To_EMA20_Pct", "LF_Daily_Dist_To_EMA200_Pct",
            "LF_Dist_To_Closest_LT_Support_Pct", "LF_Dist_To_Closest_LT_Resistance_Pct",
        ]
  
Dont forget whole/half dollar levels, micro pullbacks, and tape speed/imbalance.
Comprehensive Feature Set for Fast-Paced Momentum Trading
High-Frequency (1-Second) Features
Price Action Features

HF_Micro_Pullback_Detector - Identifies small retracements during strong trends
HF_Breakout_Strength - Measures strength of breakouts from consolidation areas
HF_Relative_Position_To_VWAP - Current price relative to intraday VWAP
HF_Dollar_Breakout_Proximity - Proximity to next whole or half-dollar price level
HF_Dollar_Breakout_Detector - Binary feature detecting break of dollar/half-dollar levels
HF_Recent_High_Test - Detection of tests of recent high prices
HF_Price_Momentum_3s - Short-term price momentum over 3 seconds
HF_Price_Momentum_5s - Short-term price momentum over 5 seconds
HF_Price_Momentum_10s - Short-term price momentum over 10 seconds
HF_Candle_Pattern_Recognition - Identification of momentum-related 1s candle patterns

Volume Features

HF_Volume_Surge_Detector - Identifies sudden spikes in trading volume
HF_Relative_Volume_1s - Current second volume compared to average second volume
HF_Volume_Acceleration - Rate of change of volume (second derivative)
HF_Price_Volume_Correlation_5s - Correlation between price and volume last 5 seconds
HF_VWAP_Divergence - Difference between VWAP and current price

Order Book/Tape Features

HF_Tape_Speed - Rate of transactions appearing on the time and sales
HF_Tape_Color_Imbalance - Ratio of buy orders (green) vs sell orders (red) on tape
HF_Order_Book_Imbalance - Difference between bid and ask sides of the order book
HF_Large_Order_Detector - Identifies unusually large orders appearing on the tape
HF_Buy_Sell_Order_Ratio - Ratio of buy to sell orders in the last second
HF_Spread_Tightening - Detection of narrowing bid-ask spreads
HF_Spread_Widening - Detection of widening bid-ask spreads
HF_Aggressive_Buyer_Pressure - Buyers hitting ask price vs limit orders
HF_Aggressive_Seller_Pressure - Sellers hitting bid price vs limit orders
HF_Bid_Ask_Size_Imbalance - Ratio of bid size to ask size
HF_Order_Flow_Delta - Net difference between aggressive buy and sell orders
HF_L2_Depth_Imbalance - Volume imbalance across multiple price levels
HF_Buyer_Exhaustion - Detection of diminishing buy pressure after surge

Medium-Frequency (1-Minute) Features
Price Action Features

MF_Candlestick_Pattern - Identification of bullish/bearish 1-minute candle patterns
MF_Bollinger_Band_Width - Width of Bollinger Bands indicating volatility
MF_Bollinger_Band_Position - Position of price relative to Bollinger Bands
MF_EMA9_Relation - Relation of price to 9-period EMA
MF_EMA20_Relation - Relation of price to 20-period EMA
MF_EMA9_20_Cross - Detection of crosses between 9 and 20 EMAs
MF_Price_Range_Expansion - Expansion of price range indicating momentum
MF_Higher_Highs_Count - Count of consecutive higher highs
MF_Higher_Lows_Count - Count of consecutive higher lows
MF_Lower_Highs_Count - Count of consecutive lower highs
MF_Lower_Lows_Count - Count of consecutive lower lows
MF_Impulse_Wave_Detection - Identification of strong directional moves
MF_Fibonacci_Level_Tests - Tests of Fibonacci retracement/extension levels
MF_MACD_Signal - MACD indicator signals
MF_MACD_Histogram_Change - Rate of change in MACD histogram
MF_RSI_Value - Current RSI value
MF_RSI_Divergence - Price/RSI divergence detection

Volume Features

MF_Volume_Profile_Peak - Identification of high volume price levels
MF_Volume_Profile_Gap - Low volume areas in volume profile
MF_Relative_Volume_1m - Current minute volume relative to average minute volume
MF_Cumulative_Volume_Delta - Net buying/selling pressure based on volume
MF_Volume_VWAP_Relation - Relation between volume and VWAP
MF_Money_Flow_Index - Money flow index indicating buying/selling pressure
MF_On_Balance_Volume_Change - Rate of change in OBV
MF_Climax_Volume_Bar - Identification of volume climax
MF_Volume_Trend_Consistency - Consistency of volume with price trend

Pattern Recognition Features

MF_Support_Resistance_Test - Tests of support/resistance levels
MF_Cup_Handle_Formation - Detection of cup and handle patterns
MF_Bull_Flag_Pattern - Detection of bull flag patterns
MF_Bear_Flag_Pattern - Detection of bear flag patterns
MF_Double_Top_Bottom - Detection of double top/bottom patterns
MF_Triangle_Pattern - Detection of triangle consolidation patterns
MF_Breakout_Failure_Risk - Risk assessment of false breakouts
MF_Momentum_Divergence - Divergence between price action and momentum indicators

Low-Frequency (5-Minute) Features
Trend Features

LF_Trend_Strength - Overall strength of the current trend
LF_Trend_Duration - Duration of current trend
LF_EMA200_Relation - Position of price relative to 200 EMA
LF_EMA9_EMA200_Relation - Relation between 9 and 200 EMAs
LF_EMA20_EMA200_Relation - Relation between 20 and 200 EMAs
LF_Impulse_Wave_Count - Count of impulse waves in current trend
LF_Pullback_Count - Count of pullbacks in current trend
LF_Average_Pullback_Depth - Average depth of pullbacks
LF_Trend_Channel_Width - Width of trend channel

Price Structure Features

LF_Key_Level_Proximity - Proximity to major support/resistance levels
LF_Daily_High_Low_Relation - Position relative to day's high/low
LF_Recent_Range_Position - Position within recent trading range
LF_Gap_Fill_Potential - Potential for filling intraday gaps
LF_Previous_Day_Close_Relation - Relation to previous day's closing price
LF_Swing_High_Low_Pattern - Pattern of recent swing highs/lows
LF_Volume_Node_Proximity - Proximity to significant volume profile nodes
LF_Price_Acceptance_Rejection - Signs of price acceptance/rejection at levels

Market Context Features

LF_Sector_Relative_Strength - Performance relative to sector
LF_Market_Correlation - Correlation with broader market indices
LF_Stock_Specific_Event_Impact - Impact of stock-specific news/events
LF_Unusual_Activity_Score - Measure of unusual price/volume activity
LF_Historical_Volatility_Percentile - Current volatility relative to historical levels
LF_Average_True_Range - Average true range indicating volatility
LF_Squeeze_Probability - Likelihood of an explosive move based on volatility contraction
LF_Extended_Move_Risk - Risk assessment of extended price moves

Static Features

Static_Float_Size - Size of the floating shares
Static_Relative_Float_Metric - Float size relative to average daily volume
Static_Short_Interest - Percentage of float sold short
Static_Historical_Squeeze_Response - Historical tendency to move explosively
Static_Dollar_Level_Significance - Historical significance of whole/half dollar levels
Static_News_Catalyst_Strength - Strength of current news catalyst
Static_Sector_Momentum - Overall momentum in the stock's sector
Static_Historical_Volatility - Historical volatility metrics

Portfolio Features

Portfolio_Current_Position_Size - Current position size as percentage of portfolio
Portfolio_Current_Position_Duration - Duration of current position
Portfolio_Unrealized_PnL - Current unrealized profit/loss
Portfolio_Max_Adverse_Excursion - Maximum adverse excursion during trade
Portfolio_Position_Type - Type of current position (long, short, flat)

This feature set is designed to capture the key elements needed for your trading style, with emphasis on:

Fast detection of momentum buildup
Recognition of whole/half dollar breakouts
Identification of micro pullbacks for entry points
Analysis of tape speed and imbalance for signal confirmation
Detection of big buyers/sellers through order book analysis
Recognition of volume surges that precede price moves

These features can be used to train your AI model to identify high-probability trading opportunities in the fast-paced, volatile low-float stock environment you're targeting.