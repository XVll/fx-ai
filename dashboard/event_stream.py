# dashboard/event_stream.py - Event streaming for detailed trading data

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import threading
import logging
from enum import Enum


class EventType(Enum):
    """Types of events that can be streamed"""

    TRADE_EXECUTION = "trade_execution"
    POSITION_UPDATE = "position_update"
    MARKET_UPDATE = "market_update"
    ORDER_BOOK_UPDATE = "order_book_update"
    ACTION_DECISION = "action_decision"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    RESET_POINT = "reset_point"


@dataclass
class TradingEvent:
    """Container for trading events"""

    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingEventStream:
    """
    Event streaming system for detailed trading data.
    Separate from metrics system to handle high-frequency detailed data.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = logging.getLogger(__name__)
        self._subscribers: List[Callable] = []
        self._event_buffer = deque(maxlen=1000)  # Keep last 1000 events
        self._subscriber_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        self._initialized = True

    def subscribe(self, callback: Callable[[TradingEvent], None]):
        """Subscribe to event stream"""
        with self._subscriber_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable):
        """Unsubscribe from event stream"""
        with self._subscriber_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Emit an event to all subscribers"""
        # Use timestamp from data if available, otherwise use current time
        timestamp = data.get("timestamp", datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        event = TradingEvent(
            event_type=event_type,
            timestamp=timestamp,
            data=data,
            metadata=metadata or {},
        )

        # Buffer event
        with self._buffer_lock:
            self._event_buffer.append(event)

        # Notify subscribers in separate threads to avoid blocking
        with self._subscriber_lock:
            for subscriber in self._subscribers:
                threading.Thread(
                    target=self._notify_subscriber,
                    args=(subscriber, event),
                    daemon=True,
                ).start()

    def _notify_subscriber(self, subscriber: Callable, event: TradingEvent):
        """Notify a single subscriber"""
        try:
            subscriber(event)
        except Exception as e:
            self.logger.error(f"Error notifying subscriber: {e}")

    def emit_trade(
        self,
        side: str,
        quantity: int,
        price: float,
        fill_price: float,
        pnl: float,
        commission: float,
        order_id: str,
        **kwargs,
    ):
        """Emit a trade execution event"""
        self.emit(
            EventType.TRADE_EXECUTION,
            {
                "side": side,
                "quantity": quantity,
                "price": price,
                "fill_price": fill_price,
                "pnl": pnl,
                "commission": commission,
                "order_id": order_id,
                **kwargs,
            },
        )

    def emit_position_update(
        self,
        side: str,
        quantity: int,
        avg_price: float,
        current_price: float,
        unrealized_pnl: float,
        realized_pnl: float,
        **kwargs,
    ):
        """Emit a position update event"""
        self.emit(
            EventType.POSITION_UPDATE,
            {
                "side": side,
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                **kwargs,
            },
        )

    def emit_market_update(
        self,
        symbol: str,
        price: float,
        bid: float,
        ask: float,
        volume: int,
        bid_size: int,
        ask_size: int,
        **kwargs,
    ):
        """Emit a market data update"""
        self.emit(
            EventType.MARKET_UPDATE,
            {
                "symbol": symbol,
                "price": price,
                "bid": bid,
                "ask": ask,
                "spread": ask - bid,
                "spread_pct": ((ask - bid) / price) * 100 if price > 0 else 0,
                "volume": volume,
                "bid_size": bid_size,
                "ask_size": ask_size,
                **kwargs,
            },
        )

    def emit_action_decision(
        self,
        action: str,
        confidence: float,
        reasoning: Dict[str, Any],
        features: Dict[str, float],
        **kwargs,
    ):
        """Emit an action decision event with reasoning"""
        self.emit(
            EventType.ACTION_DECISION,
            {
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "features": features,
                **kwargs,
            },
        )

    def get_recent_events(
        self, event_type: Optional[EventType] = None, limit: int = 100
    ) -> List[TradingEvent]:
        """Get recent events from buffer"""
        with self._buffer_lock:
            events = list(self._event_buffer)

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def clear_buffer(self):
        """Clear event buffer"""
        with self._buffer_lock:
            self._event_buffer.clear()


# Global instance
event_stream = TradingEventStream()
