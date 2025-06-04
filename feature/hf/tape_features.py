"""Tape (trade) analysis features"""

from typing import Dict, Any
from feature.feature_base import BaseFeature, FeatureConfig


class TapeImbalanceFeature(BaseFeature):
    """Buy/sell volume imbalance in 1-second window"""

    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="tape_imbalance", normalize=False)
        super().__init__(config)

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate buy/sell imbalance"""
        hf_window = market_data.get("hf_data_window", [])

        if not hf_window:
            return 0.0

        # Get trades from the last second
        trades = hf_window[-1].get("trades", []) if hf_window else []

        if not trades:
            return 0.0

        buy_volume = 0.0
        sell_volume = 0.0

        # If we have quotes, use them to classify trades
        quotes = hf_window[-1].get("quotes", [])
        last_quote = quotes[-1] if quotes else None

        for trade in trades:
            size = trade.get("size", 0)
            if size <= 0:
                continue

            # Check if trade has conditions
            conditions = trade.get("conditions", [])
            if conditions:
                if "BUY" in conditions or "B" in conditions:
                    buy_volume += size
                elif "SELL" in conditions or "S" in conditions:
                    sell_volume += size
                else:
                    # Try to infer from price
                    if last_quote:
                        price = trade.get("price")
                        bid = last_quote.get("bid_price")
                        ask = last_quote.get("ask_price")

                        if price and bid and ask:
                            mid = (bid + ask) / 2
                            if price >= ask:
                                buy_volume += size  # Hit ask = buy
                            elif price <= bid:
                                sell_volume += size  # Hit bid = sell
                            else:
                                # At mid, split evenly
                                buy_volume += size / 2
                                sell_volume += size / 2
                    else:
                        # No way to classify, split evenly
                        buy_volume += size / 2
                        sell_volume += size / 2
            else:
                # No conditions, try to infer from price vs quotes
                if last_quote:
                    price = trade.get("price")
                    bid = last_quote.get("bid_price")
                    ask = last_quote.get("ask_price")

                    if price and bid and ask:
                        if price >= ask:
                            buy_volume += size
                        elif price <= bid:
                            sell_volume += size
                        else:
                            # At mid
                            buy_volume += size / 2
                            sell_volume += size / 2
                else:
                    # No way to classify
                    buy_volume += size / 2
                    sell_volume += size / 2

        # Calculate imbalance
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0

        imbalance = (buy_volume - sell_volume) / total_volume
        return imbalance

    def get_default_value(self) -> float:
        """Default to neutral"""
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        """Already in [-1, 1] range"""
        return {}

    def get_requirements(self) -> Dict[str, Any]:
        return {"data_type": "trades", "fields": ["trades"], "lookback": 1}


class TapeAggressionRatioFeature(BaseFeature):
    """Ratio of aggressive orders (hitting bid/ask)"""

    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="tape_aggression_ratio", normalize=False)
        super().__init__(config)

    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate aggression ratio"""
        hf_window = market_data.get("hf_data_window", [])

        if not hf_window:
            return 0.0

        current_data = hf_window[-1]
        trades = current_data.get("trades", [])
        quotes = current_data.get("quotes", [])

        if not trades or not quotes:
            return 0.0

        # Get the last quote before trades
        last_quote = quotes[-1]
        bid = last_quote.get("bid_price")
        ask = last_quote.get("ask_price")

        if not bid or not ask or bid >= ask:
            return 0.0

        aggressive_buy_volume = 0.0
        aggressive_sell_volume = 0.0
        total_volume = 0.0

        for trade in trades:
            price = trade.get("price")
            size = trade.get("size", 0)

            if not price or size <= 0:
                continue

            total_volume += size

            # Classify aggression
            if price >= ask:
                aggressive_buy_volume += size
            elif price <= bid:
                aggressive_sell_volume += size
            # Trades at mid are not aggressive

        if total_volume == 0:
            return 0.0

        # Calculate net aggression ratio
        ratio = (aggressive_buy_volume - aggressive_sell_volume) / total_volume
        return ratio

    def get_default_value(self) -> float:
        """Default to neutral"""
        return 0.0

    def get_normalization_params(self) -> Dict[str, Any]:
        """Already in [-1, 1] range"""
        return {}

    def get_requirements(self) -> Dict[str, Any]:
        return {"data_type": "trades", "fields": ["trades", "quotes"], "lookback": 1}
