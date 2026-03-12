from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


TARGET_SEGMENTS = ['プライム', 'スタンダード', 'グロース']

DEFAULT_FACTOR_CONFIG: dict[str, dict[str, float | str]] = {
    'ret_5m': {'label': '直近5分の値動き', 'weight': 0.15, 'direction': 1},
    'ret_1d': {'label': '当日の値動き', 'weight': 0.20, 'direction': 1},
    'volume_ratio': {'label': '出来高の勢い', 'weight': 0.08, 'direction': 1},
    'volatility_2h': {'label': '値動きの荒さ', 'weight': 0.08, 'direction': -1},
    'rsi_14': {'label': '短期の過熱感', 'weight': 0.07, 'direction': -1},
    'roe': {'label': 'ROE', 'weight': 0.10, 'direction': 1},
    'trailing_pe': {'label': 'PERの割安感', 'weight': 0.08, 'direction': -1},
    'price_to_book': {'label': 'PBRの割安感', 'weight': 0.05, 'direction': -1},
    'dividend_yield': {'label': '配当利回り', 'weight': 0.02, 'direction': 1},
    'news_sentiment': {'label': 'ニュースの市場心理', 'weight': 0.11, 'direction': 1},
    'tdnet_sentiment': {'label': 'TDnet開示の方向感', 'weight': 0.09, 'direction': 1},
    'tdnet_disclosure_count': {'label': 'TDnet開示件数', 'weight': 0.03, 'direction': 1},
    'tdnet_earnings_revision_score': {'label': '業績修正の傾き', 'weight': 0.04, 'direction': 1},
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


def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(n) or np.isinf(n):
        return default
    return n


def _build_major_factor_sentence(effects: list[tuple[str, float]], row: pd.Series) -> str:
    top_three = effects[:3]
    factor_parts: list[str] = []
    for name, value in top_three:
        impact = '追い風' if value >= 0 else '重し'
        factor_parts.append(f'「{name}」が{impact}（寄与 {value:+.2f}）')

    score = _safe_float(row, 'score', 0.0)
    stance_text = '買い寄り' if score >= 0 else '売り寄り'

    ret_5m = _safe_float(row, 'ret_5m', 0.0)
    ret_1d = _safe_float(row, 'ret_1d', 0.0)
    news_sentiment = _safe_float(row, 'news_sentiment', 0.0)
    news_count = int(round(_safe_float(row, 'news_article_count', 0.0), 0))
    trailing_pe = _safe_float(row, 'trailing_pe', 0.0)
    price_to_book = _safe_float(row, 'price_to_book', 0.0)
    volatility_2h = _safe_float(row, 'volatility_2h', 0.0)
    tdnet_sentiment = _safe_float(row, 'tdnet_sentiment', 0.0)
    tdnet_count = int(round(_safe_float(row, 'tdnet_disclosure_count', 0.0), 0))
    tdnet_revision = _safe_float(row, 'tdnet_earnings_revision_score', 0.0)

    momentum_text = (
        f'直近5分は{ret_5m:+.2%}、当日は{ret_1d:+.2%}で、足元の値動きは'
        f"{'上向き' if (ret_5m + ret_1d) >= 0 else '下向き'}です"
    )
    news_view_text = (
        f'ニュースは{news_count}件を確認し、市場心理は{news_sentiment:+.2f} '
        f"（{'前向き' if news_sentiment >= 0 else '慎重'}）と判定しています"
    )
    valuation_text = (
        f'株価水準はPER {trailing_pe:.2f}倍、PBR {price_to_book:.2f}倍で、'
        f"{'割安感が意識されやすい' if (trailing_pe > 0 and trailing_pe <= 15 and price_to_book > 0 and price_to_book <= 1.5) else '割高感・中立感を見極めたい'}局面です"
    )
    risk_text = (
        f'直近2時間の値動きの荒さは{volatility_2h:.2%}で、'
        f"{'想定より振れやすい状態' if volatility_2h >= 0.01 else '比較的落ち着いた状態'}です"
    )

    if tdnet_count <= 0:
        disclosure_text = 'TDnetの新規開示は直近で目立たず、需給と価格推移を中心に判定しています'
    else:
        revision_view = '上方修正寄り' if tdnet_revision > 0 else ('下方修正寄り' if tdnet_revision < 0 else '中立')
        sentiment_view = '前向き' if tdnet_sentiment >= 0 else '慎重'
        disclosure_text = (
            f'TDnetでは直近{tdnet_count}件の開示を確認し、開示トーンは{sentiment_view}'
            f'（方向感 {tdnet_sentiment:+.2f}）、業績修正の傾きは{revision_view}として反映しています'
        )

    return (
        f'総合判定は{stance_text}です。'
        f"主な材料は{'、'.join(factor_parts)}です。"
        f'{momentum_text}。{news_view_text}。{valuation_text}。{risk_text}。{disclosure_text}。'
    )


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
        return pd.DataFrame(columns=['市場区分', '順位', '銘柄コード', '銘柄名', '終値', '前日比', 'スコア', 'スコアの主な要因'])

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
        return pd.DataFrame(columns=['市場区分', '順位', '銘柄コード', '銘柄名', '終値', '前日比', 'スコア', 'スコアの主な要因'])

    ranked = pd.concat(frames, ignore_index=True)
    ranked['終値'] = ranked['last_price'].round(2)
    ranked['前日比'] = (ranked['ret_1d'] * 100.0).map(lambda x: f'{x:+.2f}%')
    ranked['score'] = ranked['score'].round(2)

    return ranked[['市場区分', '順位', 'code', 'name', '終値', '前日比', 'score', 'major_factors']].rename(
        columns={
            'code': '銘柄コード',
            'name': '銘柄名',
            'score': 'スコア',
            'major_factors': 'スコアの主な要因',
        }
    )

