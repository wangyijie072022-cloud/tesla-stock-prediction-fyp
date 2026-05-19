"""Shared date helpers for TSLA market-day prediction scripts.

The project environment does not include a market-calendar package, so this
module carries the small XNYS calendar surface needed by the local scripts:
regular holidays, known ad-hoc closures in the project period, and standard
early-close sessions.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


US_MARKET_TZ = ZoneInfo("America/New_York")
US_MARKET_REGULAR_CLOSE_TIME = time(16, 0)
US_MARKET_EARLY_CLOSE_TIME = time(13, 0)
SUPPORTED_EXCHANGES = {"XNYS", "NASDAQ"}
AD_HOC_MARKET_CLOSURES = {
    Date(2025, 1, 9),  # National Day of Mourning for President Jimmy Carter.
}


def easter_date(year: int) -> Date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    lower_l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lower_l) // 451
    month = (h + lower_l - 7 * m + 114) // 31
    day = ((h + lower_l - 7 * m + 114) % 31) + 1
    return Date(year, month, day)


def observed_fixed_holiday(year: int, month: int, day: int) -> Date:
    holiday = Date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def nth_weekday(year: int, month: int, weekday: int, n: int) -> Date:
    current = Date(year, month, 1)
    days_until = (weekday - current.weekday()) % 7
    return current + timedelta(days=days_until + (n - 1) * 7)


def last_weekday(year: int, month: int, weekday: int) -> Date:
    if month == 12:
        current = Date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = Date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _validate_exchange(exchange: str) -> str:
    normalized = exchange.upper()
    if normalized not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Unsupported exchange calendar: {exchange}. Supported: {sorted(SUPPORTED_EXCHANGES)}")
    return normalized


def us_market_holidays(year: int) -> set[Date]:
    holidays = {
        observed_fixed_holiday(year, 1, 1),
        nth_weekday(year, 1, 0, 3),
        nth_weekday(year, 2, 0, 3),
        easter_date(year) - timedelta(days=2),
        last_weekday(year, 5, 0),
        observed_fixed_holiday(year, 7, 4),
        nth_weekday(year, 9, 0, 1),
        nth_weekday(year, 11, 3, 4),
        observed_fixed_holiday(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(observed_fixed_holiday(year, 6, 19))
    return holidays.union(day for day in AD_HOC_MARKET_CLOSURES if day.year == year)


def is_us_market_trading_day(day: Date, exchange: str = "XNYS") -> bool:
    _validate_exchange(exchange)
    return day.weekday() < 5 and day not in us_market_holidays(day.year)


def previous_calendar_trading_day(day: Date, exchange: str = "XNYS") -> Date:
    _validate_exchange(exchange)
    current = day
    while not is_us_market_trading_day(current, exchange):
        current -= timedelta(days=1)
    return current


def next_calendar_trading_day(day: Date, exchange: str = "XNYS") -> Date:
    _validate_exchange(exchange)
    current = day + timedelta(days=1)
    while not is_us_market_trading_day(current, exchange):
        current += timedelta(days=1)
    return current


def thanksgiving_date(year: int) -> Date:
    return nth_weekday(year, 11, 3, 4)


def is_us_market_early_close_day(day: Date, exchange: str = "XNYS") -> bool:
    _validate_exchange(exchange)
    if not is_us_market_trading_day(day, exchange):
        return False
    if day == thanksgiving_date(day.year) + timedelta(days=1):
        return True
    if day.month == 12 and day.day == 24:
        return True
    if day.month == 7 and day.day == 3 and Date(day.year, 7, 4).weekday() < 5:
        return True
    return False


def market_close_time(day: Date, exchange: str = "XNYS") -> time:
    _validate_exchange(exchange)
    if not is_us_market_trading_day(day, exchange):
        raise ValueError(f"{day.isoformat()} is not a {exchange} trading day.")
    if is_us_market_early_close_day(day, exchange):
        return US_MARKET_EARLY_CLOSE_TIME
    return US_MARKET_REGULAR_CLOSE_TIME


def market_close_datetime_utc(day: Date, exchange: str = "XNYS") -> datetime:
    close_local = datetime.combine(day, market_close_time(day, exchange), tzinfo=US_MARKET_TZ)
    return close_local.astimezone(timezone.utc)


def previous_available_date(day: Date, available_dates: list[Date]) -> Date | None:
    candidates = [available_day for available_day in available_dates if available_day <= day]
    return max(candidates) if candidates else None


def next_available_date(day: Date, available_dates: list[Date]) -> Date | None:
    candidates = [available_day for available_day in available_dates if available_day > day]
    return min(candidates) if candidates else None


def ensure_utc_datetime(value: object | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise TypeError(f"Expected datetime-like cutoff, got {type(value).__name__}.")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def get_latest_closed_trading_day(cutoff: object | None = None, exchange: str = "XNYS") -> Date:
    _validate_exchange(exchange)
    cutoff_utc = ensure_utc_datetime(cutoff)
    cutoff_ny = cutoff_utc.astimezone(US_MARKET_TZ)
    cutoff_day = cutoff_ny.date()
    if is_us_market_trading_day(cutoff_day, exchange):
        if cutoff_utc >= market_close_datetime_utc(cutoff_day, exchange):
            return cutoff_day
        return previous_calendar_trading_day(cutoff_day - timedelta(days=1), exchange)
    return previous_calendar_trading_day(cutoff_day, exchange)


def latest_closed_trading_day(cutoff: object | None = None) -> Date:
    return get_latest_closed_trading_day(cutoff, exchange="XNYS")


def closed_market_cutoff_note(cutoff: object | None, selected_day: Date, exchange: str = "XNYS") -> str:
    _validate_exchange(exchange)
    cutoff_utc = ensure_utc_datetime(cutoff)
    cutoff_ny = cutoff_utc.astimezone(US_MARKET_TZ)
    cutoff_day = cutoff_ny.date()
    if selected_day == cutoff_day:
        return ""
    if is_us_market_trading_day(cutoff_day, exchange):
        cutoff_close = market_close_datetime_utc(cutoff_day, exchange).astimezone(US_MARKET_TZ)
        return (
            f"Prediction cutoff is before {exchange} market close "
            f"({cutoff_close.strftime('%H:%M %Z')}); "
            f"using latest closed trading day: {selected_day.isoformat()}."
        )
    selected_close = market_close_datetime_utc(selected_day, exchange).astimezone(US_MARKET_TZ)
    return (
        "Prediction cutoff is before the latest requested market close or falls on a non-trading day; "
        f"using latest closed trading day: {selected_day.isoformat()} "
        f"(market close {selected_close.strftime('%H:%M %Z')})."
    )
