
from datetime import date, datetime, timedelta
from typing import Optional, Union, Protocol
import numpy as np
import pandas as pd
import holidays
import pandas_market_calendars as mcal

# --- 1. Define a Protocol for Calendars ---
# This allows you to pass ANY calendar object (custom, pandas, numpy-based)
# as long as it has these two attributes.
class MarketCalendar(Protocol):
    @property
    def days_in_year(self):
        """Usually 365 for scalar, 252 for equities."""
        pass
        
    def count_days(self, start: date, end: date):
        """Count days between start and end according to market rules."""
        pass

# --- 4. Custom Holiday Calendar for User-defined Schedules ---
class CustomHolidayCalendar:
    
    """
    Calendar that excludes weekends and a user-provided list of holidays.
    Accepts holidays as a list of strings, date, datetime, or pandas.Timestamp.
    """
       
    def __init__(self, holidays=None, days_in_year=252):
        self.holidays = self._parse_holidays(holidays) if holidays else []
        self._days_in_year = days_in_year

    @property
    def days_in_year(self) -> int:
        return self._days_in_year

    def count_days(self, start: date, end: date):
        return np.busday_count(start, end, holidays=self.holidays)

    def next_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> date:
        dt = self._parse_date(d)
        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def previous_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> date:
        dt = self._parse_date(d)
        prev_day = dt - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def is_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> bool:
        dt = self._parse_date(d)
        return bool(np.is_busday(dt, holidays=self.holidays))

    def _parse_date(self, d):
        if isinstance(d, str):
            try:
                return datetime.strptime(d, '%Y-%m-%d').date()
            except ValueError:
                return datetime.strptime(d, '%d-%m-%Y').date()
        elif isinstance(d, datetime):
            return d.date()
        elif isinstance(d, date):
            return d
        elif isinstance(d, pd.Timestamp):
            return d.date()
        else:
            raise TypeError("Date must be string, date, datetime, or pd.Timestamp")

    def _parse_holidays(self, holidays):
        # Accepts list of str/date/datetime/Timestamp, returns list of date
        parsed = []
        for h in holidays:
            parsed.append(self._parse_date(h))
        return parsed

# --- 2. A Concrete Implementation for Equity Markets ---
class WeekdayCalendar:
    """
    A simple calendar that excludes weekends using numpy.
    Assumes 252 trading days per year.
    """

    def __init__(self, holidays=None):
        self.holidays = holidays if holidays else []
        self._days_in_year = 252

    @property
    def days_in_year(self) -> int:
        return self._days_in_year

    def count_days(self, start: date, end: date):
        # np.busday_count excludes weekends and specific holidays
        return np.busday_count(start, end, holidays=self.holidays)

    def is_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> bool:
        """Return True if d is a trading day (not weekend or holiday)."""
        dt = self._parse_date(d)
        # np.is_busday returns True for business days (Mon-Fri, not in holidays)
        return bool(np.is_busday(dt, holidays=self.holidays))

    def next_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> date:
        dt = self._parse_date(d)
        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def previous_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> date:
        dt = self._parse_date(d)
        prev_day = dt - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def _parse_date(self, d):
        if isinstance(d, str):
            try:
                return datetime.strptime(d, '%Y-%m-%d').date()
            except ValueError:
                return datetime.strptime(d, '%d-%m-%Y').date()
        elif isinstance(d, datetime):
            return d.date()
        elif isinstance(d, date):
            return d
        elif isinstance(d, pd.Timestamp):
            return d.date()
        else:
            raise TypeError("Date must be string, date, datetime, or pd.Timestamp")

class PublicHolidayCalendar:
    """
    A calendar that excludes weekends AND public holidays for a specific country.
    """
    def __init__(self, region: str = 'US', days_in_year: int = 252):
        self.region = region
        self._days_in_year = days_in_year
        # We load the holiday object for the specific country
        # getattr fetches 'US', 'UK', 'DE' etc. from the holidays library
        self.holiday_loader = getattr(holidays, region)()

    @property
    def days_in_year(self) -> int:
        return self._days_in_year

    def count_days(self, start: date, end: date):
        """
        Counts business days excluding weekends and country-specific holidays.
        """
        if start >= end:
            return 0
            
        # 1. Identify all unique years between start and end to fetch relevant holidays
        years = list(range(start.year, end.year + 1))
        
        # 2. Get the list of holiday dates for these years
        # The holidays library lets us pass a list of years to populate
        holiday_list = []
        for year in years:
            # We add holidays for that year to our list
            holiday_list.extend(self.holiday_loader.get(date(year, 1, 1)) for _ in range(1)) 
            # Note: The library usage might vary slightly by version, 
            # but the standard way is simply accessing the object like a dict or using .get()
            # A cleaner way with the modern library is simply passing years to the constructor,
            # but since we initialized it once, we can just instantiate a new one or update it:
            
        # A cleaner, robust implementation to get all holidays in range:
        relevant_holidays = []
        # specific instance for the required years
        country_holidays = getattr(holidays, self.region)(years=years)
        relevant_holidays = list(country_holidays.keys())

        # 3. Use numpy to count business days, excluding the holidays we found
        return np.busday_count(start, end, holidays=relevant_holidays)
    
    def next_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> date:
        dt = self._parse_date(d)
        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def previous_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> date:
        dt = self._parse_date(d)
        prev_day = dt - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def is_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> bool:
        """Return True if d is a trading day (not weekend or holiday)."""
        dt = self._parse_date(d)
        # Get holidays for the year of dt
        country_holidays = getattr(holidays, self.region)(years=[dt.year])
        relevant_holidays = set(country_holidays.keys())
        # np.is_busday returns True for business days (Mon-Fri, not in holidays)
        return bool(np.is_busday(dt, holidays=list(relevant_holidays)))

    def _parse_date(self, d):
        if isinstance(d, str):
            try:
                return datetime.strptime(d, '%Y-%m-%d').date()
            except ValueError:
                return datetime.strptime(d, '%d-%m-%Y').date()
        elif isinstance(d, datetime):
            return d.date()
        elif isinstance(d, date):
            return d
        elif isinstance(d, pd.Timestamp):
            return d.date()
        else:
            raise TypeError("Date must be string, date, datetime, or pd.Timestamp")
    
class ExchangeCalendar:
    """
    Uses pandas_market_calendars to count strict exchange trading sessions.
    """

    def __init__(self, exchange_name: str = 'NYSE', days_in_year: int = 252):
        self.exchange_name = exchange_name
        self._days_in_year = days_in_year
        
        # Load the specific exchange calendar (e.g., 'NYSE', 'LSE', 'JPX')
        try:
            self.calendar = mcal.get_calendar(exchange_name)
        except RuntimeError:
            # Fallback or helpful error if the exchange name is wrong
            raise ValueError(f"Calendar '{exchange_name}' not found. "
                             f"Available calendars: {mcal.get_calendar_names()[:5]}...")

    @property
    def days_in_year(self) -> int:
        return self._days_in_year

    def count_days(self, start: date, end: date) -> int:
        """
        Returns the number of valid trading sessions between start and end.
        Logic: Measures the 'distance' in trading days.
        """
        if start >= end:
            return 0

        # We convert date objects to pandas Timestamp for compatibility
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # LOGIC NOTE: 
        # To calculate 'Time To Maturity' (distance), we look for valid trading 
        # days strictly *after* the start date up to and including the end date.
        # Example: Valuation=Monday, Expiry=Tuesday -> Distance = 1 trading day.
        
        schedule = self.calendar.schedule(
            start_date=start_ts + timedelta(days=1), 
            end_date=end_ts
        )
        
        # The length of the schedule DataFrame is the number of open market days
        return len(schedule)

    def next_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> date:
        dt = self._parse_date(d)
        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def previous_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> date:
        dt = self._parse_date(d)
        prev_day = dt - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def is_trading_day(self, d: Union[str, date, datetime, pd.Timestamp]) -> bool:
        """Return True if d is a trading day (open session on exchange)."""
        dt = self._parse_date(d)
        ts = pd.Timestamp(dt)
        # Get the schedule for that day
        sched = self.calendar.schedule(start_date=ts, end_date=ts)
        return not sched.empty

    def _parse_date(self, d):
        if isinstance(d, str):
            try:
                return datetime.strptime(d, '%Y-%m-%d').date()
            except ValueError:
                return datetime.strptime(d, '%d-%m-%Y').date()
        elif isinstance(d, datetime):
            return d.date()
        elif isinstance(d, date):
            return d
        elif isinstance(d, pd.Timestamp):
            return d.date()
        else:
            raise TypeError("Date must be string, date, datetime, or pd.Timestamp")