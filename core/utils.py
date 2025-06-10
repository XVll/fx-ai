from typing import List, Optional

import pendulum


# Todo : Test this function
def day_in_range(day: str, date_range: List[Optional[str]]) -> bool:
    """Check if day falls within date range using pendulum"""
    if not date_range[0] and not date_range[1]:
        return True

    day_date = pendulum.parse(day)

    pendulum.date(day_date.year, day_date.month, day_date.day)
    if date_range[0]:
        start_date =  pendulum.parse(date_range[0])
        if day_date < start_date:
            return False

    if date_range[1]:
        end_date = pendulum.parse(date_range[1])
        if day_date > end_date:
            return False

    return True
