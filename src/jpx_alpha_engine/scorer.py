from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


TARGET_SEGMENTS = ['プライム', 'スタンダード', 'グロース']

DEFAULT_FACTOR_CONFIG: dict[str, dict[str, float | str]] = {
    'ret_5m': {'label': '5分モメンタム', 'weight': 0.15, 'direction': 1},
    'ret_1d': {'label': '1日モメンタム', 'weight': 0.20, 'direction': 1},
    'volume_ratio': {'label': '出来高急増', 'weight': 0.08, 'direction': 1},
    'volatility_2h': {'label': '低ボラティリティ', 'weight': 0.08, 'direction': -1},
    'rsi_14': {'label': 'RSI(逆張り)', 'weight': 0.07, 'direction': -1},
    'roe': {'label': 'ROE', 'weight': 0.12, 'direction': 1},
    'trailing_pe': {'label': 'PER割安', 'weight': 0.08, 'direction': -1},
    'price_to_book': {'label': 'PBR割安', 'weight': 0.05, 'direction': -1},
    'dividend_yield': {'label': '配当利回り', 'weight': 0.02, 'direction': 1},
    'news_sentiment': {'label': 'ニュースセンチメント', 'weight': 0.15, 'direction': 1},
}


def get_default_factor_config() -> dict[str, dict[str, float | str]]:
    return deepcopy(DEFAULT_FACTOR_CONFIG)


def get_default_weights() -> dict[str, float]:
    return {k: float(v['weight']) for k, v in DEFAULT_FACTOR_CONFIG.items()}


def _resolve_factor_config(weight_overrides: dict[str, float] | None) -> dict[str, dict[str, float | str]]:
    config = get_default_factor_config()
    if not weight_overrides:
        return config
    for factor, weight in weight_overrides.items():
        if factor in config:
            config[factor]['weight'] = float(weight)
    return config


def _normalize_market_segment(value: object) -> str:
    text = str(value or '')
    if 'プライム' in text:
        return 'プライム'
    if 'スタンダード' in text:
        return 'スタンダード'
    if 'グロース' in text:
        return 'グロース'
    return '未分類'


def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def _build_major_factor_sentence(effects: list[tuple[str, float]], row: pd.Series) -> str:
    top_three = effects[:3]
    phrases: list[str] = []
    for name, value in top_three:
        direction_text = '押し上げ要因' if value >= 0 else '押し下げ要因'
        phrases.append(f'{name}が{value:+.2f}で{direction_text}')

    headline = '、'.join(phrases)
    ret_5m = float(row.get('ret_5m', 0.0) or 0.0)
    ret_1d = float(row.get('ret_1d', 0.0) or 0.0)
    news_sentiment = float(row.get('news_sentiment', 0.0) or 0.0)
    news_count = int(float(row.get('news_article_count', 0.0) or 0.0))
    trailing_pe = float(row.get('trailing_pe', 0.0) or 0.0)
    price_to_book = float(row.get('price_to_book', 0.0) or 0.0)
    volatility_2h = float(row.get('volatility_2h', 0.0) or 0.0)

    momentum_text = (
        f'直近5分騰落率は{ret_5m:+.2%}、1日騰落率は{ret_1d:+.2%}で、短期モメンタムは'
        f"{'上向き' if (ret_5m + ret_1d) >= 0 else '下向き'}です"
    )
    news_text = f'ニュースは{news_count}件を評価し、センチメントは{news_sentiment:+.2f}でした'
    valuation_text = f'バリュエーション面ではPER {trailing_pe:.2f}倍、PBR {price_to_book:.2f}倍の水準です'
    risk_text = (
        f'2時間ボラティリティは{volatility_2h:.2%}で、'
        f"値動きの荒さは{'高め' if volatility_2h >= 0.01 else '落ち着き気味'}と判定しました"
    )
    return f'{headline}。{momentum_text}。{news_text}。{valuation_text}。{risk_text}。'


def score_snapshot(snapshot: pd.DataFrame, weight_overrides: dict[str, float] | None = None) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot

    factor_config = _resolve_factor_config(weight_overrides)
    scored = snapshot.copy()
    scored['market_segment'] = scored['market_segment'].map(_normalize_market_segment)
    contribution_cols: list[str] = []

    for col, cfg in factor_config.items():
        raw = scored[col].replace([np.inf, -np.inf], np.nan)
        filled = raw.fillna(raw.median())
        z = _zscore(filled)
        z_col = f'z__{col}'
        scored[z_col] = z
        contrib_col = f'contrib__{col}'
        scored[contrib_col] = z * float(cfg['weight']) * float(cfg['direction'])
        contribution_cols.append(contrib_col)

    scored['score_raw'] = scored[contribution_cols].sum(axis=1)
    scored['score'] = (scored['score_raw'] * 100).round(4)
    scored['signal'] = np.where(scored['score'] >= 0, '買い候補', '売り候補')

    major_reasons: list[str] = []
    for _, row in scored.iterrows():
        effects: list[tuple[str, float]] = []
        for col, cfg in factor_config.items():
            effects.append((str(cfg['label']), float(row[f'contrib__{col}'])))
        effects.sort(key=lambda x: abs(x[1]), reverse=True)
        major_reasons.append(_build_major_factor_sentence(effects, row))

    scored['major_factors'] = major_reasons
    return scored


def build_recommendation_csv(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(columns=['市場区分', '順位', '銘柄コード', '銘柄名', '現値', '前日比', 'スコア', 'スコアの主要な要因'])

    frames: list[pd.DataFrame] = []

    for segment in TARGET_SEGMENTS:
        seg_df = scored[scored['market_segment'] == segment].copy()
        if seg_df.empty:
            continue

        buy_top = seg_df[seg_df['score'] >= 0].sort_values('score', ascending=False).head(top_k).copy().reset_index(drop=True)
        buy_top['順位'] = buy_top.index + 1
        buy_top['市場区分'] = segment

        sell_top = seg_df[seg_df['score'] < 0].sort_values('score', ascending=True).head(top_k).copy().reset_index(drop=True)
        sell_top['順位'] = sell_top.index + 1
        sell_top['市場区分'] = segment

        frames.extend([buy_top, sell_top])

    if not frames:
        return pd.DataFrame(columns=['市場区分', '順位', '銘柄コード', '銘柄名', '現値', '前日比', 'スコア', 'スコアの主要な要因'])

    ranked = pd.concat(frames, ignore_index=True)
    ranked['現値'] = ranked['last_price'].round(2)
    ranked['前日比'] = (ranked['ret_1d'] * 100.0).map(lambda x: f'{x:+.2f}%')
    ranked['score'] = ranked['score'].round(2)

    return ranked[['市場区分', '順位', 'code', 'name', '現値', '前日比', 'score', 'major_factors']].rename(
        columns={
            'code': '銘柄コード',
            'name': '銘柄名',
            'score': 'スコア',
            'major_factors': 'スコアの主要な要因',
        }
    )
