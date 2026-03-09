from __future__ import annotations

import logging

import pandas as pd
import requests

from .config import load_settings


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
LOGGER = logging.getLogger(__name__)

JPX_XLS_URL = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'


def _pick_column(df: pd.DataFrame, keywords: list[str]) -> str:
    for col in df.columns:
        name = str(col)
        if all(k in name for k in keywords):
            return str(col)
    raise ValueError(f'Column not found for keywords={keywords}. columns={list(df.columns)}')


def _normalize_market_segment(value: object) -> str:
    text = str(value or '')
    if 'プライム' in text:
        return 'プライム'
    if 'スタンダード' in text:
        return 'スタンダード'
    if 'グロース' in text:
        return 'グロース'
    return ''


def main() -> None:
    settings = load_settings()

    raw_path = settings.project_root / 'data' / '_jpx_data_j.xls'
    out_path = settings.universe_path

    LOGGER.info('Downloading JPX issue list: %s', JPX_XLS_URL)
    response = requests.get(JPX_XLS_URL, timeout=30)
    response.raise_for_status()
    raw_path.write_bytes(response.content)

    LOGGER.info('Parsing XLS: %s', raw_path)
    df = pd.read_excel(raw_path, dtype=str)
    df = df.fillna('')

    code_col = _pick_column(df, ['コード'])
    name_col = _pick_column(df, ['銘柄名'])

    market_col = None
    for candidate in [['市場', '区分'], ['市場', '第一'], ['市場']]:
        try:
            market_col = _pick_column(df, candidate)
            break
        except ValueError:
            continue

    if not market_col:
        raise ValueError('Market segment column not found in JPX file.')

    result = pd.DataFrame(
        {
            'code': df[code_col].astype(str).str.extract(r'(\d{4})', expand=False),
            'name': df[name_col].astype(str).str.strip(),
            'market_segment': df[market_col].map(_normalize_market_segment),
        }
    )

    result = result.dropna(subset=['code'])
    result = result[result['code'].str.fullmatch(r'\d{4}')]
    result = result[result['market_segment'] != '']
    result['yf_ticker'] = result['code'] + '.T'
    result = result.drop_duplicates(subset=['code']).sort_values(['market_segment', 'code']).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result[['code', 'name', 'yf_ticker', 'market_segment']].to_csv(out_path, index=False, encoding='utf-8-sig')

    summary = result['market_segment'].value_counts().to_dict()
    LOGGER.info('Universe generated: %s symbols -> %s', len(result), out_path)
    LOGGER.info('Segment counts: %s', summary)


if __name__ == '__main__':
    main()
