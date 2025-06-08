"""
Mock implementations of all trading environment interfaces.

These mocks are useful for:
- Unit testing
- Development without real dependencies
- Isolated component testing
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
import numpy as np

import sys
from pathlib import Path

from v2.simulation.interfaces import IPortfolioSimulator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v2.envs.interfaces import (
    MarketData,
    PortfolioState,
    ObservationDict,
    ActionArray,
    ActionMask,
    ActionProbabilities,
    FillDetails,
    ExecutionResult,
    ResetPoint,
    MomentumDay,
    PriceDict,
)


class MockMarketSimulator:
    """Mock implementation of IMarketSimulator for testing."""
    
    def __init__(self, seq_len_hf: int = 60, feat_dim_hf: int = 10,
                 seq_len_mf: int = 20, feat_dim_mf: int = 15,
                 seq_len_lf: int = 10, feat_dim_lf: int = 20):
        self.seq_len_hf = seq_len_hf
        self.feat_dim_hf = feat_dim_hf
        self.seq_len_mf = seq_len_mf
        self.feat_dim_mf = feat_dim_mf
        self.seq_len_lf = seq_len_lf
        self.feat_dim_lf = feat_dim_lf
        
        self.current_time = datetime(2024, 1, 1, 9, 30)
        self.current_price = 100.0
        self.step_count = 0
        self.is_initialized = False
    
    def initialize_day(self, date: datetime) -> bool:
        """Initialize simulator for trading day."""
        self.current_time = date.replace(hour=9, minute=30, second=0)
        self.current_price = 100.0 + np.random.normal(0, 5)  # Random starting price
        self.step_count = 0
        self.is_initialized = True
        return True
    
    def reset(self) -> bool:
        """Reset simulator state."""
        if not self.is_initialized:
            return False
        self.step_count = 0
        return True
    
    def step(self) -> bool:
        """Advance one time step."""
        if not self.is_initialized:
            return False
        
        self.step_count += 1
        self.current_time += timedelta(seconds=1)
        
        # Simple price random walk
        price_change = np.random.normal(0, 0.01) * self.current_price
        self.current_price = max(0.01, self.current_price + price_change)
        
        # Stop after market hours (8 PM)
        return self.current_time.hour < 20
    
    def get_current_market_data(self) -> Optional[MarketData]:
        """Get current market data."""
        if not self.is_initialized:
            return None
        
        spread = self.current_price * 0.001  # 0.1% spread
        return {
            "timestamp": self.current_time,
            "current_price": self.current_price,
            "best_bid_price": self.current_price - spread/2,
            "best_ask_price": self.current_price + spread/2,
            "volume": np.random.randint(1000, 10000),
            "open": self.current_price * (1 + np.random.normal(0, 0.01)),
            "high": self.current_price * (1 + abs(np.random.normal(0, 0.02))),
            "low": self.current_price * (1 - abs(np.random.normal(0, 0.02))),
            "close": self.current_price,
        }
    
    def get_current_features(self) -> Optional[ObservationDict]:
        """Get pre-calculated features."""
        if not self.is_initialized:
            return None
        
        # Generate random features matching expected dimensions
        return {
            "hf": np.random.randn(self.seq_len_hf, self.feat_dim_hf).astype(np.float32),
            "mf": np.random.randn(self.seq_len_mf, self.feat_dim_mf).astype(np.float32),
            "lf": np.random.randn(self.seq_len_lf, self.feat_dim_lf).astype(np.float32),
        }
    
    def set_time(self, timestamp: datetime) -> bool:
        """Set simulator to specific timestamp."""
        if not self.is_initialized:
            return False
        
        self.current_time = timestamp
        return True
    
    def get_current_time(self) -> Optional[datetime]:
        """Get current simulation time."""
        return self.current_time if self.is_initialized else None


class MockPortfolioSimulator:
    """Mock implementation of IPortfolioSimulator for testing."""
    
    def __init__(self, initial_capital: float = 25000.0,
                 portfolio_seq_len: int = 5, portfolio_feat_dim: int = 8):
        self.initial_capital = initial_capital
        self.portfolio_seq_len = portfolio_seq_len
        self.portfolio_feat_dim = portfolio_feat_dim
        
        self.cash = initial_capital
        self.positions = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.current_time = datetime.now()
    
    def reset(self, session_start: datetime) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.current_time = session_start
    
    def get_portfolio_state(self, timestamp: datetime) -> PortfolioState:
        """Get portfolio state at timestamp."""
        self.current_time = timestamp
        
        # Calculate total equity
        total_equity = self.cash + self.unrealized_pnl
        
        return {
            "timestamp": timestamp,
            "total_equity": total_equity,
            "cash": self.cash,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl_session": self.realized_pnl,
            "positions": self.positions,
            "session_metrics": {
                "total_commissions_session": 0.0,
                "total_fees_session": 0.0,
                "total_slippage_cost_session": 0.0,
            }
        }
    
    def get_portfolio_observation(self) -> ObservationDict:
        """Get portfolio observation for model."""
        # Generate mock portfolio features
        features = np.random.randn(self.portfolio_seq_len, self.portfolio_feat_dim).astype(np.float32)
        
        # Set some realistic values in first row
        features[0, 0] = self.cash / self.initial_capital  # Cash ratio
        features[0, 1] = self.unrealized_pnl / self.initial_capital  # Unrealized PnL ratio
        features[0, 2] = len(self.positions)  # Number of positions
        
        return {"features": features}
    
    def update_market_values(self, prices: PriceDict, timestamp: datetime) -> None:
        """Update portfolio with market prices."""
        self.current_time = timestamp
        
        # Update unrealized PnL based on current prices
        self.unrealized_pnl = 0.0
        for asset, position_data in self.positions.items():
            if asset in prices:
                current_price = prices[asset]
                quantity = position_data.get("quantity", 0.0)
                avg_entry = position_data.get("avg_entry_price", current_price)
                
                # Simple P&L calculation
                pnl_per_share = current_price - avg_entry
                position_pnl = pnl_per_share * quantity
                self.unrealized_pnl += position_pnl
    
    def process_fill(self, fill_details: FillDetails) -> FillDetails:
        """Process trade fill and update portfolio."""
        asset = fill_details["asset_id"]
        quantity = fill_details["executed_quantity"]
        price = fill_details["executed_price"]
        side = fill_details["order_side"]
        
        # Adjust quantity for sell orders
        if side.value == "SELL":
            quantity = -quantity
        
        # Update position
        if asset not in self.positions:
            self.positions[asset] = {
                "quantity": 0.0,
                "avg_entry_price": 0.0,
                "current_side": "FLAT",
                "unrealized_pnl": 0.0,
            }
        
        position = self.positions[asset]
        old_quantity = position["quantity"]
        new_quantity = old_quantity + quantity
        
        # Update average entry price
        if new_quantity != 0:
            if old_quantity == 0:  # Opening new position
                position["avg_entry_price"] = price
            elif (old_quantity > 0) == (quantity > 0):  # Adding to position
                total_cost = old_quantity * position["avg_entry_price"] + quantity * price
                position["avg_entry_price"] = total_cost / new_quantity
        
        position["quantity"] = new_quantity
        
        # Update cash
        cost = quantity * price
        commission = max(1.0, abs(cost) * 0.001)  # Simple commission
        self.cash -= cost + commission
        
        # Update position side
        if new_quantity > 0:
            position["current_side"] = "LONG"
        elif new_quantity < 0:
            position["current_side"] = "SHORT"
        else:
            position["current_side"] = "FLAT"
        
        # Add enriched details
        enriched_fill = dict(fill_details)
        enriched_fill.update({
            "commission": commission,
            "fees": 0.0,
            "slippage_cost_total": abs(cost) * 0.0005,  # Small slippage
        })
        
        return enriched_fill
    
    def get_trading_metrics(self) -> Dict[str, float]:
        """Get trading metrics."""
        return {
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "win_rate": 0.5,  # Mock value
            "total_trades": 0,
            "max_drawdown": 0.0,
        }


class MockExecutionSimulator:
    """Mock implementation of IExecutionSimulator for testing."""
    
    def __init__(self):
        self.np_random = np.random.default_rng()
    
    def reset(self, np_random_seed_source: Optional[np.random.Generator] = None) -> None:
        """Reset execution simulator."""
        if np_random_seed_source:
            self.np_random = np_random_seed_source
        else:
            self.np_random = np.random.default_rng()
    
    def execute_action(self, raw_action: ActionArray, market_state: MarketData,
                      portfolio_state: PortfolioState, primary_asset: str,
                      portfolio_manager: IPortfolioSimulator) -> ExecutionResult:
        """Execute trading action."""
        action_type = int(raw_action[0])  # 0=HOLD, 1=BUY, 2=SELL
        position_size = int(raw_action[1])  # 0=25%, 1=50%, 2=75%, 3=100%
        
        # Decode action
        action_decode_result = {
            "action_type": ["HOLD", "BUY", "SELL"][action_type],
            "position_size": (position_size + 1) * 0.25,
            "is_valid": True,
        }
        
        fill_details = None
        
        # Execute if not HOLD
        if action_type != 0:  # Not HOLD
            current_price = market_state["current_price"]
            cash = portfolio_state["cash"]
            
            # Calculate order size
            if action_type == 1:  # BUY
                max_shares = cash / current_price
                order_size = max_shares * action_decode_result["position_size"]
                order_side = "BUY"
            else:  # SELL
                # For now, assume we can short sell
                max_shares = cash / current_price  # Simplified
                order_size = max_shares * action_decode_result["position_size"]
                order_side = "SELL"
            
            if order_size > 0:
                # Add some execution slippage
                execution_price = current_price * (1 + self.np_random.normal(0, 0.001))
                
                fill_details = {
                    "asset_id": primary_asset,
                    "executed_quantity": order_size,
                    "executed_price": execution_price,
                    "order_side": order_side,
                    "order_type": "MARKET",
                    "fill_timestamp": market_state["timestamp"],
                }
        
        return {
            "action_decode_result": action_decode_result,
            "fill_details": fill_details,
        }


class MockRewardCalculator:
    """Mock implementation of IRewardCalculator for testing."""
    
    def __init__(self):
        self.last_components = {}
    
    def reset(self) -> None:
        """Reset reward calculator state."""
        self.last_components = {}
    
    def calculate(self, portfolio_state_before_action: PortfolioState,
                 portfolio_state_after_action_fills: PortfolioState,
                 portfolio_state_next_t: PortfolioState,
                 market_state_at_decision: MarketData,
                 market_state_next_t: Optional[MarketData],
                 decoded_action: Dict[str, any],
                 fill_details_list: List[FillDetails],
                 terminated: bool,
                 truncated: bool,
                 termination_reason: Optional[any]) -> float:
        """Calculate step reward."""
        
        # Simple reward based on equity change
        equity_before = portfolio_state_before_action["total_equity"]
        equity_after = portfolio_state_next_t["total_equity"]
        equity_change = equity_after - equity_before
        
        # Reward components
        pnl_reward = equity_change / 1000.0  # Scale down
        action_penalty = -0.001 if decoded_action["action_type"] != "HOLD" else 0.0
        
        total_reward = pnl_reward + action_penalty
        
        self.last_components = {
            "pnl_reward": pnl_reward,
            "action_penalty": action_penalty,
            "total_reward": total_reward,
        }
        
        return total_reward
    
    def get_last_reward_components(self) -> Dict[str, float]:
        """Get reward component breakdown."""
        return self.last_components.copy()


class MockActionMask:
    """Mock implementation of IActionMask for testing."""
    
    def is_action_valid(self, action_idx: int, portfolio_state: PortfolioState,
                       market_state: MarketData) -> bool:
        """Check if action is valid."""
        # Convert linear index to action type and size
        action_type = action_idx // 4
        position_size = action_idx % 4
        
        # HOLD is always valid
        if action_type == 0:
            return True
        
        # Check if we have enough cash for BUY
        if action_type == 1:  # BUY
            cash = portfolio_state["cash"]
            price = market_state["current_price"]
            return cash > price * 10  # Need at least 10 shares worth
        
        # For SELL, assume always valid (can short)
        return True
    
    def get_action_mask(self, portfolio_state: PortfolioState,
                       market_state: MarketData) -> ActionMask:
        """Get action mask for current state."""
        mask = np.ones(12, dtype=bool)  # 3 actions x 4 sizes = 12
        
        for i in range(12):
            mask[i] = self.is_action_valid(i, portfolio_state, market_state)
        
        return mask
    
    def mask_action_probabilities(self, action_probs: ActionProbabilities,
                                 portfolio_state: PortfolioState,
                                 market_state: MarketData) -> ActionProbabilities:
        """Apply action masking to probabilities."""
        mask = self.get_action_mask(portfolio_state, market_state)
        
        # Mask invalid actions
        masked_probs = action_probs * mask
        
        # Renormalize
        prob_sum = masked_probs.sum()
        if prob_sum > 0:
            masked_probs = masked_probs / prob_sum
        else:
            # If all invalid, uniform over valid actions
            masked_probs = mask / mask.sum()
        
        return masked_probs
    
    def get_valid_actions(self, portfolio_state: PortfolioState, 
                         market_state: MarketData) -> List[str]:
        """Get list of valid action descriptions."""
        valid_actions = []
        mask = self.get_action_mask(portfolio_state, market_state)
        
        action_names = ["HOLD", "BUY", "SELL"]
        size_names = ["25%", "50%", "75%", "100%"]
        
        for i, is_valid in enumerate(mask):
            if is_valid:
                action_type = i // 4
                position_size = i % 4
                desc = f"{action_names[action_type]}_{size_names[position_size]}"
                valid_actions.append(desc)
        
        return valid_actions
    
    def get_action_description(self, action_idx: int) -> str:
        """Get action description."""
        action_type = action_idx // 4
        position_size = action_idx % 4
        
        action_names = ["HOLD", "BUY", "SELL"]
        size_names = ["25%", "50%", "75%", "100%"]
        
        return f"{action_names[action_type]}_{size_names[position_size]}"


class MockDataManager:
    """Mock implementation of IDataManager for testing."""
    
    def get_reset_points(self, symbol: str, date: datetime) -> List[ResetPoint]:
        """Get reset points for symbol and date."""
        base_time = date.replace(hour=9, minute=30, second=0)
        
        reset_points = []
        for i in range(4):  # 4 reset points throughout the day
            timestamp = base_time + timedelta(hours=i*2)
            reset_points.append({
                "timestamp": timestamp,
                "activity_score": 0.5 + np.random.random() * 0.5,
                "combined_score": 0.3 + np.random.random() * 0.7,
                "max_duration_hours": 2.0,
                "reset_type": "momentum",
                "price": 100.0 + np.random.normal(0, 5),
                "volume": np.random.randint(1000, 10000),
            })
        
        return reset_points
    
    def get_momentum_days(self, symbol: str, min_activity: float = 0.0) -> List[MomentumDay]:
        """Get momentum days for symbol."""
        momentum_days = []
        
        # Generate 5 days of momentum data
        base_date = datetime(2024, 1, 1)
        for i in range(5):
            date = base_date + timedelta(days=i)
            activity = 0.3 + np.random.random() * 0.7
            
            if activity >= min_activity:
                momentum_days.append({
                    "symbol": symbol,
                    "date": date,
                    "activity_score": activity,
                    "max_intraday_move": 0.05 + np.random.random() * 0.15,
                    "volume_multiplier": 1.0 + np.random.random() * 2.0,
                })
        
        return momentum_days