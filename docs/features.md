self.static_feature_names: List[str] = [
S_Time_Of_Day_Sin - Time encoding (morning vs afternoon behavior)
S_Time_Of_Day_Cos - Time encoding complement"Previous_Day_Postmarket_Gap_Pct",
S_Market_Session_Type - Pre-market, regular, post-market
]

self.hf_feature_names: List[str] = [
"HF_1s_Price_Velocity"  Rate of price change over the last 1 second
"HF_1s_Price_Acceleration"  # Change in price velocity over the last 1 second

"HF_1s_Volume_Velocity",  # Rate of volume change over the last 1 second
"HF_1s_Volume_Acceleration", # Change in volume velocity over the last 1 second
"HF_Tape_1s_Imbalance", # Ratio of buy vs sell volume in the last second
"HF_Tape_1s_Aggression_Ratio", # Orders hitting bid vs ask 

"HF_Quote_1s_Spread_Compression", # Change in bid-ask spread over the last second
"HF_Quote_1s_Quote_Imbalance", # Ratio of bid vs ask volume in the last second
]

self.mf_feature_names: List[str] = [
"MF_1m_Price_Velocity", 
"MF_5m_Price_Velocity", 
"MF_1m_Price_Acceleration", 
"MF_5m_Price_Acceleration",
"MF_1m_Volume_Velocity", 
"MF_5m_Volume_Velocity",
"MF_1m_Volume_Acceleration",
"MF_5m_Volume_Acceleration",
"MF_1m_Dist_To_EMA9",
"MF_1m_Dist_To_EMA20",
"MF_5m_Dist_To_EMA9",
"MF_5m_Dist_To_EMA20",

"MF_1m_Position_Current_Candle", # Position in current 1-minute candle range
"MF_5m_Position_In_CurrentCandle", # Position in current 5-minute candle range
"MF_1m_Position_In_PreviousCandle", # Position in previous 1-minute candle range
"MF_5m_Position_In_PreviousCandle", # Position in previous 5-minute candle range
"MF_1m_BodySize_Rel", 
"MF_5m_BodySize_Rel",
"MF_1m_UpperWick_Rel",
"MF_1m_LowerWick_Rel",
"MF_5m_UpperWick_Rel",
"MF_5m_LowerWick_Rel",
 
"MF_1m_Swing_High_Dist", 
"MF_1m_Swing_Low_Dist", 
"MF_5m_Swing_High_Dist",
"MF_5m_Swing_Low_Dist",

]
self.lf_feature_names: List[str] = [
"LF_Position_In_Daily_Range", 
"LF_Position_In_PrevDay_Range", 
"LF_Price_Change_From_Prev_Close",
"LF_Dist_To_Closest_LT_Support", 
"LF_Dist_To_Closest_LT_Resistance",
"LF_Whole_Dollar_Proximity", 
"LF_Half_Dollar_Proximity",  

]

self.portfolio_feature_names: List[str] = [
"Portfolio_Current_Position_Size",  # Current position size as percentage of portfolio
"Portfolio_Average_Price",  # Average entry price of current position
"Portfolio_Unrealized_PnL",  # Current unrealized profit/loss
"Portfolio_Time_In_Position",  # Duration of current position
"Portfolio_Max_Adverse_Excursion", # Maximum adverse excursion during trade