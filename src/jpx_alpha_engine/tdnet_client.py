from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


LOGGER = logging.getLogger(__name__)

TDNET_FEED_CANDIDATES = [
    'https://www.release.tdnet.info/inbs/I_list_001.xml',
    'https://www.release.tdnet.info/inbs/I_list_001_2026.xml',
    'https://www.release.tdnet.info/inbs/I_main_00.html',
]
REQUEST_TIMEOUT_SECONDS = 8
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}
JST = timezone(timedelta(hours=9))

POSITIVE_TERMS = {
    '上方修正', '増配', '自社株買い', '黒字転換', '受注拡大', '提携', '成長',
}
NEGATIVE_TERMS = {
    '下方修正', '減配', '赤字転落', '特別損失', '不正', '訴訟', '希薄化',
}


@dataclass(frozen=True)
class TdnetDisclosure:
    code: str
    title: str
    published_at: datetime
    source: str


def _default_features() -> dict[str, float]:
    return {
        'tdnet_sentiment': 0.0,
        'tdnet_disclosure_count': 0.0,
        'tdnet_positive_count': 0.0,
        'tdnet_negative_count': 0.0,
        'tdnet_earnings_revision_score': 0.0,
    }


class TdnetClient:
    def __init__(
        self,
        cache_path: Path,
        lookback_days: int = 7,
        max_items: int = 1200,
    ) -> None:
        self.cache_path = cache_path
        self.lookback_days = max(1, int(lookback_days))
        self.max_items = max(100, int(max_items))
        self.cache = self._load_cache()
        self.disclosures: list[TdnetDisclosure] = []

    def build_feature_map(self, security_codes: list[str]) -> dict[str, dict[str, float]]:
        code_set = {str(code) for code in security_codes if str(code)}
        if not code_set:
            return {}

        self.disclosures = self._load_recent_disclosures()

        feature_map = {code: _default_features() for code in code_set}
        for item in self.disclosures:
            if item.code not in feature_map:
                continue

            features = feature_map[item.code]
            features['tdnet_disclosure_count'] += 1.0

            text = item.title.strip()
            pos = sum(1 for term in POSITIVE_TERMS if term in text)
            neg = sum(1 for term in NEGATIVE_TERMS if term in text)
            if pos > neg:
                features['tdnet_positive_count'] += 1.0
            elif neg > pos:
                features['tdnet_negative_count'] += 1.0

            if '上方修正' in text:
                features['tdnet_earnings_revision_score'] += 1.0
            if '下方修正' in text:
                features['tdnet_earnings_revision_score'] -= 1.0

        for code, features in feature_map.items():
            count = features['tdnet_disclosure_count']
            if count > 0:
                net = features['tdnet_positive_count'] - features['tdnet_negative_count']
                features['tdnet_sentiment'] = max(-1.0, min(1.0, net / count))
                features['tdnet_earnings_revision_score'] = max(-1.0, min(1.0, features['tdnet_earnings_revision_score'] / count))

        return feature_map

    def _load_recent_disclosures(self) -> list[TdnetDisclosure]:
        fetched = self._fetch_disclosures()
        if fetched:
            self.cache['disclosures'] = [
                {
                    'code': d.code,
                    'title': d.title,
                    'published_at': d.published_at.isoformat(),
                    'source': d.source,
                }
                for d in fetched
            ]
            self._save_cache()
            return fetched

        cached_items = self.cache.get('disclosures', [])
        restored = self._parse_cached_disclosures(cached_items)
        if restored:
            LOGGER.warning('TDnet live fetch failed; using cached disclosures (%s items).', len(restored))
        return restored

    def _fetch_disclosures(self) -> list[TdnetDisclosure]:
        all_items: list[TdnetDisclosure] = []
        for url in TDNET_FEED_CANDIDATES:
            try:
                resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
                resp.raise_for_status()
                parsed = self._parse_response(url, resp.text)
                all_items.extend(parsed)
                if all_items:
                    break
            except requests.RequestException as exc:
                LOGGER.warning('TDnet fetch failed: %s (%s)', url, exc)

        if not all_items:
            return []

        cutoff = datetime.now(JST) - timedelta(days=self.lookback_days)
        filtered = [d for d in all_items if d.published_at >= cutoff]
        filtered.sort(key=lambda x: x.published_at, reverse=True)

        deduped: list[TdnetDisclosure] = []
        seen: set[tuple[str, str, str]] = set()
        for d in filtered:
            key = (d.code, d.title, d.published_at.isoformat())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(d)
            if len(deduped) >= self.max_items:
                break

        LOGGER.info('TDnet disclosures fetched: %s items', len(deduped))
        return deduped

    def _parse_response(self, source_url: str, text: str) -> list[TdnetDisclosure]:
        if source_url.endswith('.xml'):
            return self._parse_xml(source_url, text)
        return self._parse_html(source_url, text)

    def _parse_xml(self, source_url: str, text: str) -> list[TdnetDisclosure]:
        soup = BeautifulSoup(text, 'xml')
        items = soup.find_all(['item', 'entry'])

        disclosures: list[TdnetDisclosure] = []
        for item in items:
            title = self._node_text(item, ['title'])
            description = self._node_text(item, ['description', 'summary'])
            pub_text = self._node_text(item, ['pubDate', 'published', 'updated', 'date'])
            raw_text = f'{title} {description}'.strip()
            code = self._extract_code(raw_text)
            if not code:
                continue

            published_at = self._parse_datetime(pub_text)
            disclosures.append(
                TdnetDisclosure(
                    code=code,
                    title=title or raw_text[:120],
                    published_at=published_at,
                    source=source_url,
                )
            )
        return disclosures

    def _parse_html(self, source_url: str, text: str) -> list[TdnetDisclosure]:
        soup = BeautifulSoup(text, 'html.parser')

        disclosures: list[TdnetDisclosure] = []
        for a in soup.select('a[href]'):
            title = a.get_text(' ', strip=True)
            if not title:
                continue

            code = self._extract_code(title)
            if not code:
                continue

            row_text = ' '.join(a.parent.get_text(' ', strip=True).split()) if a.parent else title
            published_at = self._parse_datetime(row_text)
            disclosures.append(
                TdnetDisclosure(
                    code=code,
                    title=title,
                    published_at=published_at,
                    source=source_url,
                )
            )
        return disclosures

    def _parse_cached_disclosures(self, rows: object) -> list[TdnetDisclosure]:
        if not isinstance(rows, list):
            return []

        restored: list[TdnetDisclosure] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get('code') or '')
            title = str(row.get('title') or '')
            source = str(row.get('source') or 'cache')
            published_at = self._parse_datetime(str(row.get('published_at') or ''))
            if not code or not title:
                continue
            restored.append(
                TdnetDisclosure(
                    code=code,
                    title=title,
                    published_at=published_at,
                    source=source,
                )
            )

        cutoff = datetime.now(JST) - timedelta(days=self.lookback_days)
        return [d for d in restored if d.published_at >= cutoff]

    def _load_cache(self) -> dict[str, object]:
        try:
            if self.cache_path.exists():
                with self.cache_path.open('r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    loaded.setdefault('disclosures', [])
                    return loaded
        except Exception as exc:
            LOGGER.warning('Failed to load TDnet cache: %s', exc)
        return {'disclosures': []}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open('w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _node_text(item: BeautifulSoup, names: list[str]) -> str:
        for name in names:
            node = item.find(name)
            if node and node.get_text(strip=True):
                return node.get_text(' ', strip=True)
        return ''

    @staticmethod
    def _extract_code(text: str) -> str | None:
        patterns = [
            r'[\[\(\{【（]\s*([1-9]\d{3})\s*[\]\)\}】）]',
            r'(?:証券コード|コード)[:：\s]*([1-9]\d{3})',
            r'\b([1-9]\d{3})\b',
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if not m:
                continue
            code = m.group(1)
            if len(code) == 4:
                return code
        return None

    @staticmethod
    def _parse_datetime(text: str) -> datetime:
        text = (text or '').strip()
        if not text:
            return datetime.now(JST)

        candidates = [
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d %H:%M',
            '%Y.%m.%d %H:%M',
            '%Y-%m-%d',
            '%Y/%m/%d',
        ]

        for fmt in candidates:
            try:
                dt = datetime.strptime(text, fmt)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=JST)
                return dt.astimezone(JST)
            except ValueError:
                continue

        m = re.search(r'(20\d{2})[\./-](\d{1,2})[\./-](\d{1,2})(?:\s+(\d{1,2}):(\d{2}))?', text)
        if m:
            year, month, day, hh, mm = m.groups()
            hour = int(hh) if hh is not None else 0
            minute = int(mm) if mm is not None else 0
            return datetime(int(year), int(month), int(day), hour, minute, tzinfo=JST)

        return datetime.now(JST)

