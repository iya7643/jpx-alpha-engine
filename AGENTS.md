# AGENTS.md

## プロジェクト概要
- リポジトリ: `C:/Users/iya22/git/jpx-alpha-engine`
- 目的: 日本株の買い/売り候補を日次で算出し、`output/recommendations.csv` と Web で可視化する。
- 実行基盤: Python (`uv`), PostgreSQL (Docker), Web (nginx), scheduler (APScheduler)

## 主要コンポーネント
- `src/jpx_alpha_engine/collector.py`
  - 市場データ収集（yfinance 5分足 + 日足fallback）
  - ニュース収集（yfinance + kabutan）
  - TDnet適時開示の特徴量収集（開示トーン/件数/業績修正）
- `src/jpx_alpha_engine/scorer.py`
  - 因子zスコア合成で `score` 算出
  - 市場区分（プライム/スタンダード/グロース）ごとに買い/売りTOP N選定
- `src/jpx_alpha_engine/pipeline.py`
  - 学習 → 収集 → スコア → CSV保存 → DB保存 の本体
- `src/jpx_alpha_engine/learning.py`
  - 予測の正誤評価と因子重み更新
  - 評価は「predicted_at + horizon 到達後の最初の営業日終値」で判定（最新値ではない）
- `src/jpx_alpha_engine/db.py`
  - DB I/O、信頼度算出用統計（スコア帯別勝率）
- `web/index.html`
  - `output/recommendations.csv` を5分おきに再読込して表示

## 起動・実行
- ローカル実行:
  - `uv sync`
  - `uv run python -m src.jpx_alpha_engine.run_once`
- Docker一括起動:
  - `docker compose up -d --build`
  - Web: `http://localhost:8081/web/index.html`

## 重要な仕様
- スケジューラ: 毎日 16:30 JST（`run_scheduler.py`）
- 予測 horizon: デフォルト 1440分（1日）
- 学習の1回あたり上限: `JPX_LEARNING_MAX_PREDICTIONS_PER_RUN`（デフォルト 500）
- 信頼度列:
  - `model_predictions` 実績から `|score|` を50刻みバケットで勝率推定
  - バケット不足時は buy/sell 全体勝率へフォールバック

## 環境変数（主なもの）
- `JPX_DATABASE_URL`
- `JPX_SCHEDULE_HOUR`, `JPX_SCHEDULE_MINUTE`, `JPX_RUN_ON_START`
- `JPX_PREDICTION_HORIZON_MINUTES`, `JPX_LEARNING_RATE`
- `JPX_LEARNING_MAX_PREDICTIONS_PER_RUN`
- `JPX_SUPPRESS_YF_WARNINGS`
- `JPX_TDNET_LOOKBACK_DAYS`, `JPX_TDNET_MAX_ITEMS`, `JPX_TDNET_CACHE_PATH`

## DBテーブル（主要）
- `technical_history_daily`: 日足OHLCV履歴
- `technical_snapshots`: 収集スナップショット
- `model_predictions`: 予測・評価結果
- `factor_weights`: 学習済み因子重み

## 既知の注意点
- 学習評価は `technical_history_daily` が埋まっていないと `skipped_no_price` が増える。
  - 対策: `backfill_technical` 実行 + 日次実行で継続蓄積。
- 旧CSV列名 `スコアの主要な要因` と新列名 `スコアの主な要因` が混在する可能性がある。
  - `web/index.html` は両方を読めるフォールバック実装済み。
- 売り信頼度が極端に低い場合、相場局面/評価対象期間の偏りを疑うこと。

## 変更時のガイドライン
- まず `python -m compileall src/jpx_alpha_engine` で構文確認。
- 信頼度ロジックを変更したら `model_predictions` の集計SQLを再確認。
- 画面列追加時は `colspan` と CSS幅指定（`nth-child`）の整合を必ず確認。

