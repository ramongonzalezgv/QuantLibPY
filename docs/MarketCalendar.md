# MarketCalendar Guide

## Overview

The `MarketCalendar` module provides flexible calendar implementations for handling different types of trading day calculations in financial applications. It accounts for weekends, holidays, and exchange-specific trading sessions when calculating time-to-maturity for option pricing.

## Why MarketCalendars Matter

When pricing financial derivatives, accurate time measurement is critical. Different markets have different trading schedules:
- **Weekdays vs. Weekends**: Markets don't trade on weekends
- **Public Holidays**: Markets close on country-specific holidays
- **Exchange Hours**: Different exchanges have different hours and holidays

For example, if you're pricing an option expiring on a Tuesday but it's currently Monday, the time-to-maturity should be 1 trading day, not 1 calendar day.

## Calendar Protocol

All calendar implementations follow a standard protocol with two key methods:

```python
class MarketCalendar(Protocol):
    @property
    def days_in_year(self) -> int:
        """Usually 365 for scalar, 252 for equities."""
        pass
        
    def count_days(self, start: date, end: date) -> int:
        """Count days between start and end according to market rules."""
        pass
```

This protocol allows you to pass any calendar object that implements these methods to the pricing engines.

## Available Calendar Implementations

### 1. WeekdayCalendar

The simplest calendar that only excludes weekends.

**Use Case**: When you only need to exclude Saturdays and Sundays, and don't need country-specific holiday handling.

**Features**:
- Assumes 252 trading days per year
- Excludes weekends automatically using NumPy
- Optional custom holidays list

**Example**:
```python
from datetime import date
from src.utils.MarketCalendar import WeekdayCalendar

# Create calendar
calendar = WeekdayCalendar()

# Count trading days between two dates
start = date(2025, 1, 1)
end = date(2025, 1, 31)
trading_days = calendar.count_days(start, end)

# Check if a specific day is a trading day
is_trading = calendar.is_trading_day(date(2025, 1, 15))

# Find next/previous trading day
next_day = calendar.next_trading_day(date(2025, 1, 10))
prev_day = calendar.previous_trading_day(date(2025, 1, 10))
```

### 2. CustomHolidayCalendar

Excludes weekends and a custom list of holidays you provide.

**Use Case**: When you need to exclude specific holidays that aren't covered by standard country calendars, or when creating a custom trading schedule.

**Features**:
- Accepts holidays as strings, dates, datetimes, or pandas Timestamps
- Flexible days_in_year parameter (default: 252)
- Automatic date format parsing

**Example**:
```python
from src.utils.MarketCalendar import CustomHolidayCalendar

# Create calendar with custom holidays
holidays = ['2025-01-01', '2025-12-25', date(2025, 7, 4)]
calendar = CustomHolidayCalendar(holidays=holidays, days_in_year=252)

# Use it for calculations
trading_days = calendar.count_days(date(2025, 1, 1), date(2025, 12, 31))

# Check trading status
is_trading = calendar.is_trading_day('2025-01-01')  # False (holiday)
is_trading = calendar.is_trading_day('2025-01-02')  # True (if Thursday)
```

### 3. PublicHolidayCalendar

Excludes weekends and country-specific public holidays.

**Use Case**: When you need proper handling of national holidays for a specific country.

**Features**:
- Supports any country that has holiday data in the `holidays` library
- Automatically fetches correct holidays for any year
- Examples: 'US', 'UK', 'DE', 'JP', 'FR', etc.

**Supported Countries**: Check the [Python holidays library](https://github.com/vacanza/python-holidays) for a complete list of supported countries.

**Example**:
```python
from src.utils.MarketCalendar import PublicHolidayCalendar

# US calendar (includes Thanksgiving, Christmas, etc.)
us_calendar = PublicHolidayCalendar(region='US')
trading_days_us = us_calendar.count_days(date(2025, 1, 1), date(2025, 12, 31))

# UK calendar (includes Bank Holidays)
uk_calendar = PublicHolidayCalendar(region='UK')
is_trading = uk_calendar.is_trading_day('2025-12-25')  # False (Christmas)

# German calendar
de_calendar = PublicHolidayCalendar(region='DE')
```

### 4. ExchangeCalendar

Uses actual exchange trading sessions from `pandas_market_calendars`.

**Use Case**: When you need the most accurate trading day calculations based on actual exchange schedules, including market hours.

**Features**:
- Reflects real exchange trading sessions
- Handles exchange-specific holidays and trading hours
- Supports major exchanges globally

**Supported Exchanges**: 
- `NYSE` (New York Stock Exchange)
- `LSE` (London Stock Exchange)
- `JPX` (Japan Exchange)
- `TSE` (Tokyo Stock Exchange)
- `CME` (Chicago Mercantile Exchange)
- And many others (see `pandas_market_calendars.get_calendar_names()`)

**Example**:
```python
from src.utils.MarketCalendar import ExchangeCalendar

# NYSE calendar (actual exchange sessions)
nyse = ExchangeCalendar(exchange_name='NYSE', days_in_year=252)
trading_days = nyse.count_days(date(2025, 1, 1), date(2025, 12, 31))

# LSE calendar
lse = ExchangeCalendar(exchange_name='LSE')
is_trading = lse.is_trading_day('2025-01-01')  # False (New Year)

# Find next trading day for option settlement
next_trading = nyse.next_trading_day(date(2025, 12, 26))
```

## Common Methods Across All Calendars

All calendar implementations provide these methods:

### `count_days(start: date, end: date) -> int`
Returns the number of trading days between two dates (exclusive of start, inclusive of end).

```python
calendar = WeekdayCalendar()
days = calendar.count_days(date(2025, 1, 1), date(2025, 1, 31))
# Returns: 21 (number of weekdays in January 2025)
```

### `is_trading_day(d: Union[str, date, datetime, pd.Timestamp]) -> bool`
Checks if a specific day is a trading day.

```python
is_trading = calendar.is_trading_day('2025-01-04')  # String
is_trading = calendar.is_trading_day(date(2025, 1, 4))  # date object
is_trading = calendar.is_trading_day(pd.Timestamp('2025-01-04'))  # Pandas Timestamp
```

### `next_trading_day(d: Union[str, date, datetime, pd.Timestamp]) -> date`
Returns the next trading day after the given date.

```python
# If Jan 4 is Saturday, next_trading_day returns Jan 6 (Monday)
next_day = calendar.next_trading_day('2025-01-04')
```

### `previous_trading_day(d: Union[str, date, datetime, pd.Timestamp]) -> date`
Returns the previous trading day before the given date.

```python
# If Jan 3 is Friday and Jan 2 is a holiday, previous_trading_day returns Jan 1
prev_day = calendar.previous_trading_day('2025-01-03')
```

## Date Input Flexibility

All methods accept dates in multiple formats:
- **String**: `'2025-01-15'` (YYYY-MM-DD) or `'15-01-2025'` (DD-MM-YYYY)
- **date object**: `date(2025, 1, 15)`
- **datetime object**: `datetime(2025, 1, 15, 10, 30)`
- **pandas Timestamp**: `pd.Timestamp('2025-01-15')`

## Usage in Option Pricing

Market calendars are used throughout the pricing system to calculate accurate time-to-maturity:

```python
from datetime import date
from src.utils.MarketCalendar import PublicHolidayCalendar
from src.valuation.OptionValuationContext import OptionValuationContext
from src.engines.AnalyticalEngine import AnalyticalEngine
from src.products.EuropeanOption import EuropeanOption

# Set up calendar
calendar = PublicHolidayCalendar(region='US')

# Create valuation context with calendar
context = OptionValuationContext(
    spot_price=100.0,
    risk_free_rate=0.05,
    market_calendar=calendar  # Pass calendar here
)

# The pricing engine uses the calendar to calculate time-to-maturity
engine = AnalyticalEngine()

option = EuropeanOption(
    strike_price=105.0,
    expiry_date=date(2025, 6, 15),
    option_type='call'
)

# Time-to-maturity is calculated using the market calendar
price = engine.price(option, context)
```

## Best Practices

### 1. **Choose the Right Calendar**
- **WeekdayCalendar**: Simple scripts, no holiday considerations
- **CustomHolidayCalendar**: Unique holiday schedules
- **PublicHolidayCalendar**: Most use cases with standard holidays
- **ExchangeCalendar**: Production systems requiring exchange accuracy

### 2. **Consistency**
Use the same calendar across your entire valuation workflow to ensure consistent time calculations.

### 3. **Date Formats**
While flexible, standardize on one format in your code (e.g., always use `date` objects).

### 4. **Calendar Performance**
- `WeekdayCalendar` and `CustomHolidayCalendar` are fastest (NumPy-based)
- `PublicHolidayCalendar` is very efficient (holiday lookup once per year)
- `ExchangeCalendar` may be slower if querying many dates (but more accurate)

### 5. **Handling Errors**
Invalid date formats raise `TypeError`. Handle gracefully:

```python
try:
    is_trading = calendar.is_trading_day('invalid-date')
except TypeError as e:
    print(f"Invalid date format: {e}")
```

## Examples

### Example 1: Calculate Days to Option Expiry
```python
from datetime import date
from src.utils.MarketCalendar import PublicHolidayCalendar

calendar = PublicHolidayCalendar(region='US')
today = date(2025, 1, 15)
expiry = date(2025, 6, 20)

days_to_expiry = calendar.count_days(today, expiry)
print(f"Trading days until expiry: {days_to_expiry}")
```

### Example 2: Check if Settlement Day is Valid
```python
from datetime import date
from src.utils.MarketCalendar import ExchangeCalendar

nyse = ExchangeCalendar(exchange_name='NYSE')
settlement_date = date(2025, 1, 1)  # New Year

if not nyse.is_trading_day(settlement_date):
    settlement_date = nyse.next_trading_day(settlement_date)
    print(f"Market closed. New settlement date: {settlement_date}")
```

### Example 3: Schedule Option Exercises
```python
from datetime import date, timedelta
from src.utils.MarketCalendar import CustomHolidayCalendar

# Create calendar with company-specific holidays
company_holidays = ['2025-07-04', '2025-12-25', '2025-01-01']
calendar = CustomHolidayCalendar(holidays=company_holidays)

# Find first trading day of next week
today = date(2025, 1, 10)  # Friday
first_trading_day_next_week = calendar.next_trading_day(today)
```

## Integration with the Option Pricing System

The MarketCalendar is integrated into:
- **OptionValuationContext**: Stores the calendar for consistent time calculations
- **Pricing Engines**: Use the calendar to calculate time-to-maturity
- **Greeks Calculators**: Require accurate time measurement for Greeks calculations

See `docs/valuationContext_ENG.md` and `docs/architecture.md` for more information on how calendars fit into the overall system.
