# jpx-alpha-engine

日本株を対象に、1日1回テクニカル/ファンダメンタルズ/ニュース感情を収集し、売買候補の上位銘柄をCSV出力するエンジンです。`microsoft/qlib` を初期化して実行します。

## セットアップ (uv)

```bash
cd C:/Users/iya22/git/jpx-alpha-engine
uv sync
copy .env.example .env
```

## PostgreSQL を Docker で起動

```bash
docker compose up -d
```

デフォルト接続先:
- `postgresql://jpx:jpxpass@localhost:5432/jpx_alpha`

## 過去数年分のテクニカルデータを蓄積

```bash
uv run python -m src.jpx_alpha_engine.backfill_technical
```

- `JPX_BACKFILL_YEARS` 年分の日足OHLCVを `technical_history_daily` テーブルへ保存します。

## 実行 (1日1回)

```bash
uv run python -m src.jpx_alpha_engine.run_scheduler
```

毎回の実行で以下を実施します。
- 最新特徴量を収集して `technical_snapshots` に保存
- 予測（買い/売り）を `model_predictions` に保存
- 期限到来した予測の正誤を判定して学習
- 学習済み重みで最新スコアを算出し `output/recommendations.csv` を更新

CSV: `output/recommendations.csv`（買い候補上位N件 + 売り候補上位N件。Nは `JPX_TOP_K`）

## 画面表示

```bash
uv run python -m http.server 8080
```

[http://localhost:8080/web/index.html](http://localhost:8080/web/index.html)

## 学習ロジックの概要

- 予測時に因子の `z` 値と寄与度を保存
- `JPX_PREDICTION_HORIZON_MINUTES` 経過後に実現リターンで正誤判定
- `JPX_LEARNING_RATE` を使って因子重みを更新し `factor_weights` に保存
- 次回スコア算出から学習後の重みを適用

## 補足

- qlibの地域定数に日本専用リージョンがないため `REG_US` で初期化しています。
- ニュースは `yfinance` と `kabutan.jp` の見出しを収集し、ポジ/ネガ辞書でセンチメント化します。



## ユニバースを増やす（JPX自動生成）

```bash
uv run python -m src.jpx_alpha_engine.build_universe_from_jpx
```

- 東証プライム銘柄を優先して `data/universe_jp.csv` を再生成します。
- その後に `uv run python -m src.jpx_alpha_engine.backfill_technical` を再実行してください。
