# jpx-alpha-engine

日本株を対象に、1日1回テクニカル/ファンダメンタルズ/ニュース感情/TDnet開示を収集し、売買候補の上位銘柄をCSV出力するエンジンです。`microsoft/qlib` を初期化して実行します。

## セットアップ (uv)

```bash
cd C:/Users/iya22/git/jpx-alpha-engine
uv sync
copy .env.example .env
```

## Docker で一括起動（推奨）

`.env` を読んで起動します。

```bash
docker compose up -d --build
```

これで以下が起動します。
- PostgreSQL (`localhost:5432`)
- スケジューラ（毎日 16:30 JST に更新）
- Web (`http://localhost:8081/web/index.html`)

デフォルト動作:
- `JPX_RUN_ON_START=false` の場合、起動時実行なし
- 以降は毎日 `16:30`（`JPX_TIMEZONE=Asia/Tokyo`）で実行

主な設定値:
- 実行時刻: `JPX_SCHEDULE_HOUR`, `JPX_SCHEDULE_MINUTE`, `JPX_TIMEZONE`
- 出力件数: `JPX_TOP_K`
- 学習判定までの待ち時間: `JPX_PREDICTION_HORIZON_MINUTES`
- 学習率: `JPX_LEARNING_RATE`
- TDnet取り込み: `JPX_TDNET_LOOKBACK_DAYS`, `JPX_TDNET_MAX_ITEMS`, `JPX_TDNET_CACHE_PATH`

## ログ確認

```bash
docker compose logs -f scheduler
```

## 手動実行（必要時）

```bash
uv run python -m src.jpx_alpha_engine.run_once
```

## 過去数年分のテクニカルデータを蓄積

```bash
uv run python -m src.jpx_alpha_engine.backfill_technical
```

- `JPX_BACKFILL_YEARS` 年分の日足OHLCVを `technical_history_daily` テーブルへ保存します。

## 学習ロジックの概要

- 予測時に因子の `z` 値と寄与度を保存
- `JPX_PREDICTION_HORIZON_MINUTES` 経過後に実現リターンで正誤判定
- `JPX_LEARNING_RATE` を使って因子重みを更新し `factor_weights` に保存
- 次回スコア算出から学習後の重みを適用

## 補足

- qlibの地域定数に日本専用リージョンがないため `REG_US` で初期化しています。
- ニュースは `yfinance` と `kabutan.jp` の見出しを収集し、ポジ/ネガ辞書でセンチメント化します。
- TDnetの適時開示データを取り込み、開示トーン・開示件数・業績修正の傾きをスコアと学習に反映します。
- Web画面は `output/recommendations.csv` を5分ごとに再読み込みします。
- `JPX_FAILURE_THRESHOLD` 回連続で取得失敗した銘柄は、`data/universe_failures.json` に記録したうえで `data/universe_jp.csv` から自動除外します。
- 自動除外した銘柄を戻す場合は、`data/universe_jp.csv` と `data/universe_failures.json` を手動で修正してください。
