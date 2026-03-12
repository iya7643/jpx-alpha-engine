from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from .db import save_daily_technical_history
from .tdnet_client import TdnetClient


LOGGER = logging.getLogger(__name__)

POSITIVE_TERMS = {
    'upgrade', 'upbeat', 'strong', 'beat', 'growth', 'record', 'surge', 'bullish',
    'buy', 'outperform', 'raise', 'profit jump', '増益', '上方修正', '最高益', '好調', '堅調', '買い', '上昇',
}
NEGATIVE_TERMS = {
    'downgrade', 'weak', 'miss', 'decline', 'drop', 'fall', 'bearish', 'sell',
    'underperform', 'cut', 'loss', '減益', '下方修正', '赤字', '不振', '売り', '下落',
}

KABUTAN_STOCK_NEWS_URL = 'https://kabutan.jp/stock/news?code={code}'
REQUEST_TIMEOUT_SECONDS = 8
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}


@dataclass(frozen=True)
class UniverseRow:
    code: str
    name: str
    yf_ticker: str
    market_segment: str


def load_universe(universe_path: str) -> list[UniverseRow]:
    df = pd.read_csv(universe_path, dtype={'code': str, 'name': str, 'yf_ticker': str})
    required = {'code', 'name', 'yf_ticker'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in universe file: {missing}')

    if 'market_segment' not in df.columns:
        df['market_segment'] = '未分類'

    return [
        UniverseRow(
            code=row.code,
            name=row.name,
            yf_ticker=row.yf_ticker,
            market_segment=str(row.market_segment),
        )
        for row in df.itertuples(index=False)
    ]


def _configure_yfinance_logging(suppress: bool) -> None:
    if not suppress:
        return
    # Hide noisy yfinance per-symbol errors (e.g. possibly delisted) from normal logs.
    logging.getLogger('yfinance').setLevel(logging.CRITICAL + 1)


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return float('nan')
        x = float(value)
        if math.isfinite(x):
            return x
        return float('nan')
    except (TypeError, ValueError):
        return float('nan')


def _build_intraday_proxy_from_daily(daily_1d: pd.DataFrame) -> pd.DataFrame:
    if daily_1d.empty:
        return pd.DataFrame()

    proxy = daily_1d[['Close', 'Volume']].copy()
    proxy = proxy.dropna(subset=['Close'])
    if proxy.empty:
        return pd.DataFrame()

    if 'Volume' not in proxy.columns:
        proxy['Volume'] = 0.0

    proxy.index = pd.to_datetime(proxy.index, utc=True)
    return proxy


def _calc_technical_features(intraday_5m: pd.DataFrame, daily_1d: pd.DataFrame) -> dict[str, float]:
    if intraday_5m.empty:
        return {'ret_5m': np.nan, 'volume_ratio': np.nan, 'volatility_2h': np.nan, 'rsi_14': np.nan, 'ret_1d': np.nan}

    close = intraday_5m['Close'].astype(float)
    volume = intraday_5m['Volume'].astype(float)

    ret_5m = np.nan
    if len(close) >= 2 and close.iloc[-2] != 0:
        ret_5m = close.iloc[-1] / close.iloc[-2] - 1

    volume_ratio = np.nan
    vol_window = volume.tail(13)
    if len(vol_window) >= 2:
        baseline = vol_window.iloc[:-1].mean()
        if baseline and baseline > 0:
            volume_ratio = vol_window.iloc[-1] / baseline

    rets = close.pct_change()
    volatility_2h = rets.tail(24).std(ddof=0)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs.iloc[-1])) if len(rs) else np.nan

    ret_1d = np.nan
    if not daily_1d.empty and len(daily_1d) >= 2:
        prev = float(daily_1d['Close'].iloc[-2])
        latest = float(daily_1d['Close'].iloc[-1])
        if prev != 0:
            ret_1d = latest / prev - 1

    return {
        'ret_5m': _safe_float(ret_5m),
        'volume_ratio': _safe_float(volume_ratio),
        'volatility_2h': _safe_float(volatility_2h),
        'rsi_14': _safe_float(rsi_14),
        'ret_1d': _safe_float(ret_1d),
    }


def _extract_fundamentals(ticker: yf.Ticker) -> dict[str, float]:
    info = ticker.info or {}
    return {
        'market_cap': _safe_float(info.get('marketCap')),
        'trailing_pe': _safe_float(info.get('trailingPE')),
        'price_to_book': _safe_float(info.get('priceToBook')),
        'roe': _safe_float(info.get('returnOnEquity')),
        'dividend_yield': _safe_float(info.get('dividendYield')),
    }


def _score_text_sentiment(text: str) -> int:
    normalized = re.sub(r'\s+', ' ', text.lower())
    pos = sum(1 for t in POSITIVE_TERMS if t in normalized)
    neg = sum(1 for t in NEGATIVE_TERMS if t in normalized)
    return pos - neg


def _fetch_kabutan_news_titles(code: str, limit: int = 20) -> list[str]:
    try:
        response = requests.get(KABUTAN_STOCK_NEWS_URL.format(code=code), headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    text_lines = [line.strip() for line in soup.get_text('\n').splitlines() if line.strip()]
    pattern = re.compile(r'^\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}\s+\S+\s+(.+)$')

    titles: list[str] = []
    for line in text_lines:
        m = pattern.match(line)
        if not m:
            continue
        title = m.group(1).strip()
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= limit:
            return titles

    if titles:
        return titles

    ignore_words = {'トップ', 'チャート', '適時開示', '株価', 'ニュース'}
    for a in soup.select('a[href]'):
        href = str(a.get('href') or '')
        text = a.get_text(strip=True)
        if len(text) < 8 or text in ignore_words:
            continue
        if '/news/' not in href and '/stock/news' not in href:
            continue
        if text in titles:
            continue
        titles.append(text)
        if len(titles) >= limit:
            break
    return titles


def _load_daily_history(ticker: yf.Ticker, known_trade_date: object | None) -> pd.DataFrame:
    if known_trade_date is None:
        return ticker.history(period='30d', interval='1d', auto_adjust=False)

    start = (pd.Timestamp(known_trade_date) - pd.Timedelta(days=10)).date().isoformat()
    daily_1d = ticker.history(start=start, interval='1d', auto_adjust=False)
    if daily_1d.empty:
        return ticker.history(period='30d', interval='1d', auto_adjust=False)
    return daily_1d


def _load_intraday_history(ticker: yf.Ticker) -> pd.DataFrame:
    intraday_5m = ticker.history(period='1d', interval='5m', auto_adjust=False)
    usable = intraday_5m.dropna(subset=['Close']) if not intraday_5m.empty else intraday_5m
    if len(usable) >= 24:
        return intraday_5m

    return ticker.history(period='2d', interval='5m', auto_adjust=False)


def _extract_news_sentiment(ticker: yf.Ticker, code: str) -> dict[str, float]:
    scored_values: list[float] = []

    try:
        news_items = ticker.news or []
    except Exception:
        news_items = []

    for item in news_items[:20]:
        title = str(item.get('title', '') or '')
        summary = str(item.get('summary', '') or '')
        text = f'{title} {summary}'.strip()
        if not text:
            continue
        raw_score = _score_text_sentiment(text)
        scored_values.append(max(-5, min(5, raw_score)) / 5.0)

    for title in _fetch_kabutan_news_titles(code, limit=20):
        raw_score = _score_text_sentiment(title)
        scored_values.append(max(-5, min(5, raw_score)) / 5.0)

    if not scored_values:
        return {'news_sentiment': 0.0, 'news_article_count': 0.0}

    return {'news_sentiment': float(np.mean(scored_values)), 'news_article_count': float(len(scored_values))}


def collect_market_snapshot(
    universe: list[UniverseRow],
    suppress_yfinance_warnings: bool = True,
    tdnet_cache_path: str | None = None,
    tdnet_lookback_days: int = 7,
    tdnet_max_items: int = 1200,
    db_url: str | None = None,
    latest_daily_dates: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    _configure_yfinance_logging(suppress_yfinance_warnings)

    tdnet_client = TdnetClient(
        cache_path=(Path(tdnet_cache_path) if tdnet_cache_path else Path('data/tdnet_cache.json')),
        lookback_days=tdnet_lookback_days,
        max_items=tdnet_max_items,
    )
    tdnet_feature_map = tdnet_client.build_feature_map([u.code for u in universe])

    rows: list[dict[str, object]] = []
    success_codes: set[str] = set()
    failed_codes: set[str] = set()

    total = len(universe)
    LOGGER.info('Starting market snapshot collection: %s symbols', total)

    for idx, item in enumerate(universe, start=1):
        try:
            LOGGER.info('[%s/%s] collecting %s (%s)', idx, total, item.code, item.name)

            ticker = yf.Ticker(item.yf_ticker)
            known_trade_date = (latest_daily_dates or {}).get(item.code)
            daily_1d = _load_daily_history(ticker, known_trade_date)
            if db_url and not daily_1d.empty:
                save_daily_technical_history(db_url, item.code, daily_1d)
            intraday_5m = _load_intraday_history(ticker)

            data_source = 'intraday_5m'
            if intraday_5m.empty:
                intraday_5m = _build_intraday_proxy_from_daily(daily_1d)
                data_source = 'daily_fallback'

            if intraday_5m.empty:
                failed_codes.add(item.code)
                LOGGER.warning('[%s/%s] no price data for %s -> failed', idx, total, item.code)
                continue

            intraday_last_close = float(intraday_5m['Close'].iloc[-1])
            latest_ts = intraday_5m.index[-1]

            latest_daily = daily_1d.tail(1)
            if latest_daily.empty:
                daily_trade_date = pd.Timestamp(latest_ts).date()
                daily_open = np.nan
                daily_high = np.nan
                daily_low = np.nan
                daily_close = latest_close
                daily_adj_close = latest_close
                daily_volume = np.nan
            else:
                drow = latest_daily.iloc[-1]
                daily_trade_date = pd.Timestamp(latest_daily.index[-1]).date()
                daily_open = _safe_float(drow.get('Open'))
                daily_high = _safe_float(drow.get('High'))
                daily_low = _safe_float(drow.get('Low'))
                daily_close = _safe_float(drow.get('Close'))
                daily_adj_close = _safe_float(drow.get('Adj Close', drow.get('Close')))
                daily_volume = _safe_float(drow.get('Volume'))

            latest_close = daily_close if not np.isnan(daily_close) else intraday_last_close

            tdnet_features = tdnet_feature_map.get(item.code, {
                'tdnet_sentiment': 0.0,
                'tdnet_disclosure_count': 0.0,
                'tdnet_positive_count': 0.0,
                'tdnet_negative_count': 0.0,
                'tdnet_earnings_revision_score': 0.0,
            })

            rows.append(
                {
                    'timestamp': latest_ts,
                    'code': item.code,
                    'name': item.name,
                    'yf_ticker': item.yf_ticker,
                    'market_segment': item.market_segment,
                    'data_source': data_source,
                    'last_price': latest_close,
                    'daily_trade_date': daily_trade_date,
                    'daily_open': daily_open,
                    'daily_high': daily_high,
                    'daily_low': daily_low,
                    'daily_close': daily_close,
                    'daily_adj_close': daily_adj_close,
                    'daily_volume': daily_volume,
                    **_calc_technical_features(intraday_5m, daily_1d),
                    **_extract_fundamentals(ticker),
                    **_extract_news_sentiment(ticker, item.code),
                    **tdnet_features,
                }
            )
            success_codes.add(item.code)

            if data_source == 'daily_fallback':
                LOGGER.warning('[%s/%s] %s used daily fallback', idx, total, item.code)

            if idx % 20 == 0 or idx == total:
                LOGGER.info(
                    'Progress: %s/%s completed (success=%s, failed=%s)',
                    idx,
                    total,
                    len(success_codes),
                    len(failed_codes),
                )
        except Exception as exc:
            failed_codes.add(item.code)
            LOGGER.warning('[%s/%s] %s failed with exception: %s', idx, total, item.code, exc)

    if not rows:
        LOGGER.warning('Snapshot collection finished with 0 rows.')
        return pd.DataFrame(), success_codes, failed_codes

    snapshot = pd.DataFrame(rows)
    snapshot['timestamp'] = pd.to_datetime(snapshot['timestamp'], utc=True).dt.tz_convert('Asia/Tokyo')

    fallback_count = int((snapshot['data_source'] == 'daily_fallback').sum())
    if fallback_count:
        LOGGER.info('Daily fallback used for %s symbols', fallback_count)

    LOGGER.info(
        'Snapshot collection completed: rows=%s success=%s failed=%s (tdnet_items=%s)',
        len(snapshot),
        len(success_codes),
        len(failed_codes),
        len(tdnet_client.disclosures),
    )
    return snapshot, success_codes, failed_codes


