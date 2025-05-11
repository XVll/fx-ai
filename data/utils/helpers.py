from datetime import datetime
from typing import Union

import pandas as pd


def ensure_timezone_aware(dt_input: Union[datetime, str], is_end_time: bool = False) -> pd.Timestamp:
    """Converts input datetime or string to a timezone-aware pandas Timestamp (UTC)."""
    if isinstance(dt_input, str):
        is_date_only_str = False
        # Basic check for "YYYY-MM-DD" like format without time
        if len(dt_input) <= 10 and dt_input.count(':') == 0 and dt_input.count('-') >= 2:
            try:
                # Further ensure it's parsable as a date
                datetime.strptime(dt_input.split(' ')[0], '%Y-%m-%d')  # Will raise ValueError if not date
                is_date_only_str = True
            except ValueError:
                is_date_only_str = False  # Not a simple date string

        if is_date_only_str:
            if is_end_time:
                # For end_time, set to the very end of the day
                ts = pd.Timestamp(f"{dt_input} 23:59:59.999999999")
            else:
                # For start_time, set to the beginning of the day
                ts = pd.Timestamp(f"{dt_input} 00:00:00.000000000")
        else:  # It's a string presumed to have time, or a more complex format
            ts = pd.Timestamp(dt_input)
    elif isinstance(dt_input, datetime):
        ts = pd.Timestamp(dt_input)
    else:
        self.logger.error(f"Unsupported type for dt_input: {type(dt_input)}. Expected datetime or str.")
        raise TypeError(f"Unsupported type for dt_input: {type(dt_input)}. Expected datetime or str.")

    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')
