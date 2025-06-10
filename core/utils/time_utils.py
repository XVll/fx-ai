"""
Time and Date utilities for consistent handling across the codebase.
Uses Pendulum exclusively for all date/time operations.

Key principles:
1. No try/catch needed - functions handle any reasonable input
2. Clear separation between date-only and datetime operations
3. Simple one-line conversions for UTC/ET
4. Consistent formatting
"""

import pendulum
from pendulum import DateTime, Date
from typing import Union, Optional, List

# Timezone constants
ET = "America/New_York"
UTC = "UTC"

# Format constants
DATE_FORMAT = "YYYY-MM-DD"
DATETIME_FORMAT = "YYYY-MM-DD HH:mm:ss"
DATETIME_FORMAT_TZ = "YYYY-MM-DD HH:mm:ss zz"
TIME_FORMAT = "HH:mm:ss"
FILENAME_DATE_FORMAT = "YYYYMMDD"
DISPLAY_DATE_FORMAT = "MMM DD, YYYY"
DISPLAY_DATETIME_FORMAT = "MMM DD, YYYY h:mm A"


# ============== Date-Only Operations ==============

def to_date(anything: Union[str, Date, DateTime, None]) -> Optional[Date]:
    """
    Convert any input to a pendulum Date object.
    
    Args:
        anything: String in any format, Date, DateTime, or None
        
    Returns:
        Date object or None if input is None or invalid
        
    Examples:
        >>> to_date("2024-01-15")
        Date(2024, 1, 15)
        >>> to_date("01/15/2024")
        Date(2024, 1, 15)
        >>> to_date("Jan 15, 2024")
        Date(2024, 1, 15)
        >>> to_date(pendulum.now())
        Date(2024, 1, 15)
    """
    if anything is None:
        return None
    if isinstance(anything, Date):
        return anything
    if isinstance(anything, DateTime):
        return anything.date()
    try:
        return pendulum.parse(str(anything)).date()
    except:
        return None


def today() -> Date:
    """Get today's date."""
    return pendulum.today().date()


def yesterday() -> Date:
    """Get yesterday's date."""
    return today().subtract(days=1)


def tomorrow() -> Date:
    """Get tomorrow's date."""
    return today().add(days=1)


def format_date(date: Optional[Date], format: str = DATE_FORMAT) -> str:
    """
    Format a date object to string.
    
    Args:
        date: Date object to format
        format: Format string (default: YYYY-MM-DD)
        
    Returns:
        Formatted date string or empty string if date is None
    """
    if date is None:
        return ""
    return date.format(format)


def date_range(start: Union[str, Date], end: Union[str, Date]) -> List[Date]:
    """
    Generate a list of dates between start and end (inclusive).
    
    Args:
        start: Start date
        end: End date
        
    Returns:
        List of Date objects
        
    Example:
        >>> date_range("2024-01-01", "2024-01-03")
        [Date(2024, 1, 1), Date(2024, 1, 2), Date(2024, 1, 3)]
    """
    start_date = to_date(start)
    end_date = to_date(end)
    
    if start_date is None or end_date is None:
        return []
    
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current = current.add(days=1)
    return dates


def is_weekend(date: Union[str, Date]) -> bool:
    """Check if a date is Saturday or Sunday."""
    date_obj = to_date(date)
    if date_obj is None:
        return False
    return date_obj.day_of_week in (pendulum.SATURDAY, pendulum.SUNDAY)


def is_weekday(date: Union[str, Date]) -> bool:
    """Check if a date is Monday through Friday."""
    return not is_weekend(date)


# ============== DateTime Operations with Timezone ==============

def to_datetime(anything: Union[str, DateTime, None], tz: str = UTC) -> Optional[DateTime]:
    """
    Convert any input to a pendulum DateTime object.
    
    Args:
        anything: String, DateTime, or None
        tz: Default timezone if not specified in input (default: UTC)
        
    Returns:
        DateTime object or None if input is None or invalid
    """
    if anything is None:
        return None
    if isinstance(anything, DateTime):
        return anything
    try:
        dt = pendulum.parse(str(anything))
        if dt.timezone is None:
            dt = pendulum.parse(str(anything), tz=tz)
        return dt
    except:
        return None


def to_et(anything: Union[str, DateTime, None]) -> Optional[DateTime]:
    """
    Convert any datetime to Eastern Time.
    Assumes UTC if no timezone specified.
    
    Args:
        anything: String or DateTime
        
    Returns:
        DateTime in ET or None
        
    Example:
        >>> to_et("2024-01-15T14:30:00")  # Assumes UTC
        DateTime(2024, 1, 15, 9, 30, 0, tzinfo=Timezone('America/New_York'))
    """
    dt = to_datetime(anything, tz=UTC)
    if dt is None:
        return None
    return dt.in_timezone(ET)


def to_utc(anything: Union[str, DateTime, None]) -> Optional[DateTime]:
    """
    Convert any datetime to UTC.
    
    Args:
        anything: String or DateTime
        
    Returns:
        DateTime in UTC or None
    """
    dt = to_datetime(anything)
    if dt is None:
        return None
    return dt.in_timezone(UTC)


def now_et() -> DateTime:
    """Get current time in Eastern Time."""
    return pendulum.now(ET)


def now_utc() -> DateTime:
    """Get current time in UTC."""
    return pendulum.now(UTC)


def format_datetime(dt: Optional[DateTime], format: str = DATETIME_FORMAT_TZ) -> str:
    """
    Format a datetime object to string.
    
    Args:
        dt: DateTime object to format
        format: Format string (default includes timezone)
        
    Returns:
        Formatted datetime string or empty string if dt is None
    """
    if dt is None:
        return ""
    return dt.format(format)


# ============== Market Hours Utilities ==============

def get_market_open_et(date: Union[str, Date]) -> DateTime:
    """Get market open time (9:30 AM ET) for a given date."""
    date_obj = to_date(date)
    if date_obj is None:
        return None
    return pendulum.datetime(date_obj.year, date_obj.month, date_obj.day, 9, 30, 0, tz=ET)


def get_market_close_et(date: Union[str, Date]) -> DateTime:
    """Get market close time (4:00 PM ET) for a given date."""
    date_obj = to_date(date)
    if date_obj is None:
        return None
    return pendulum.datetime(date_obj.year, date_obj.month, date_obj.day, 16, 0, 0, tz=ET)


def get_pre_market_open_et(date: Union[str, Date]) -> DateTime:
    """Get pre-market open time (4:00 AM ET) for a given date."""
    date_obj = to_date(date)
    if date_obj is None:
        return None
    return pendulum.datetime(date_obj.year, date_obj.month, date_obj.day, 4, 0, 0, tz=ET)


def get_after_hours_close_et(date: Union[str, Date]) -> DateTime:
    """Get after-hours close time (8:00 PM ET) for a given date."""
    date_obj = to_date(date)
    if date_obj is None:
        return None
    return pendulum.datetime(date_obj.year, date_obj.month, date_obj.day, 20, 0, 0, tz=ET)


def is_market_hours(dt: Union[str, DateTime]) -> bool:
    """
    Check if given time is during regular market hours (9:30 AM - 4:00 PM ET).
    
    Args:
        dt: DateTime to check
        
    Returns:
        True if during market hours
    """
    et_time = to_et(dt)
    if et_time is None or is_weekend(et_time.date()):
        return False
    
    market_open = get_market_open_et(et_time.date())
    market_close = get_market_close_et(et_time.date())
    
    return market_open <= et_time < market_close


def is_pre_market(dt: Union[str, DateTime]) -> bool:
    """Check if given time is during pre-market hours (4:00 AM - 9:30 AM ET)."""
    et_time = to_et(dt)
    if et_time is None or is_weekend(et_time.date()):
        return False
    
    pre_market_open = get_pre_market_open_et(et_time.date())
    market_open = get_market_open_et(et_time.date())
    
    return pre_market_open <= et_time < market_open


def is_after_hours(dt: Union[str, DateTime]) -> bool:
    """Check if given time is during after-hours trading (4:00 PM - 8:00 PM ET)."""
    et_time = to_et(dt)
    if et_time is None or is_weekend(et_time.date()):
        return False
    
    market_close = get_market_close_et(et_time.date())
    after_hours_close = get_after_hours_close_et(et_time.date())
    
    return market_close <= et_time < after_hours_close


# ============== Convenience Functions ==============

def days_between(start: Union[str, Date], end: Union[str, Date]) -> int:
    """Calculate number of days between two dates."""
    start_date = to_date(start)
    end_date = to_date(end)
    if start_date is None or end_date is None:
        return 0
    return (end_date - start_date).days


def add_trading_days(date: Union[str, Date], days: int) -> Date:
    """
    Add trading days (weekdays only) to a date.
    
    Args:
        date: Starting date
        days: Number of trading days to add (can be negative)
        
    Returns:
        Resulting date
    """
    current = to_date(date)
    if current is None:
        return None
    
    days_to_add = abs(days)
    direction = 1 if days > 0 else -1
    
    while days_to_add > 0:
        current = current.add(days=direction)
        if is_weekday(current):
            days_to_add -= 1
    
    return current


def parse_market_timestamp(timestamp: str, date: Union[str, Date]) -> DateTime:
    """
    Parse a market timestamp (HH:MM:SS) for a given date, returning ET time.
    
    Args:
        timestamp: Time string like "09:30:00" or "14:45:30"
        date: Date for the timestamp
        
    Returns:
        DateTime in ET
        
    Example:
        >>> parse_market_timestamp("14:30:00", "2024-01-15")
        DateTime(2024, 1, 15, 14, 30, 0, tzinfo=Timezone('America/New_York'))
    """
    date_obj = to_date(date)
    if date_obj is None:
        return None
    
    time_parts = timestamp.split(":")
    hour = int(time_parts[0])
    minute = int(time_parts[1]) if len(time_parts) > 1 else 0
    second = int(time_parts[2]) if len(time_parts) > 2 else 0
    
    return pendulum.datetime(date_obj.year, date_obj.month, date_obj.day, 
                            hour, minute, second, tz=ET)


# ============== Date Range Checking ==============

def day_in_range(day: Union[str, Date], date_range: List[Optional[str]]) -> bool:
    """Check if day falls within date range.
    
    Args:
        day: Either a string date or pendulum Date object
        date_range: List of [start_date, end_date] as strings
        
    Returns:
        bool: True if day is within range, False otherwise
    """
    if not date_range[0] and not date_range[1]:
        return True

    day_date = to_date(day)
    if day_date is None:
        return False

    if date_range[0]:
        start_date = to_date(date_range[0])
        if start_date and day_date < start_date:
            return False

    if date_range[1]:
        end_date = to_date(date_range[1])
        if end_date and day_date > end_date:
            return False

    return True


# ============== Validation Helpers ==============

def is_valid_trading_day(date: Union[str, Date]) -> bool:
    """
    Check if a date is a valid trading day (weekday, not a holiday).
    Note: This is a simple check. For accurate holiday checking,
    integrate with a market calendar.
    """
    return is_weekday(date)


def ensure_trading_day(date: Union[str, Date], direction: str = "next") -> Date:
    """
    Ensure the date is a trading day. If not, find the nearest one.
    
    Args:
        date: Date to check
        direction: "next" or "previous"
        
    Returns:
        Nearest trading day
    """
    current = to_date(date)
    if current is None:
        return None
    
    while not is_valid_trading_day(current):
        if direction == "next":
            current = current.add(days=1)
        else:
            current = current.subtract(days=1)
    
    return current