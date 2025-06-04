# data/utils/helpers.py
from datetime import datetime
from typing import Union
import pandas as pd


def ensure_timezone_aware(
    dt_input: Union[datetime, str], is_end_time: bool = False
) -> pd.Timestamp:
    """
    Converts input datetime or string to a timezone-aware pandas Timestamp (UTC).

    Args:
        dt_input: Input datetime or string
        is_end_time: If True and the input is date-only, sets time to end of day

    Returns:
        Timezone-aware pandas Timestamp
    """
    if isinstance(dt_input, str):
        # Check if string contains only date part
        date_only = len(dt_input) <= 10 and dt_input.count(":") == 0

        if date_only:
            # For end_time, set to end of day
            if is_end_time:
                dt_input = f"{dt_input} 23:59:59.999999"
            else:
                dt_input = f"{dt_input} 00:00:00.000000"

        # Parse string to Timestamp
        ts = pd.Timestamp(dt_input)
    elif isinstance(dt_input, datetime):
        # Convert datetime to Timestamp
        ts = pd.Timestamp(dt_input)
    else:
        raise TypeError(
            f"Unsupported type: {type(dt_input)}. Expected datetime or str."
        )

    # Ensure timezone is UTC
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    elif str(ts.tzinfo).upper() != "UTC":
        ts = ts.tz_convert("UTC")

    return ts
