CREATE TABLE IF NOT EXISTS technical_history_daily (
  code TEXT NOT NULL,
  trade_date DATE NOT NULL,
  open_price DOUBLE PRECISION,
  high_price DOUBLE PRECISION,
  low_price DOUBLE PRECISION,
  close_price DOUBLE PRECISION,
  adj_close DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  PRIMARY KEY (code, trade_date)
);

CREATE TABLE IF NOT EXISTS technical_snapshots (
  id BIGSERIAL PRIMARY KEY,
  captured_at TIMESTAMPTZ NOT NULL,
  code TEXT NOT NULL,
  name TEXT NOT NULL,
  last_price DOUBLE PRECISION NOT NULL,
  feature_json JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS model_predictions (
  id BIGSERIAL PRIMARY KEY,
  predicted_at TIMESTAMPTZ NOT NULL,
  code TEXT NOT NULL,
  name TEXT NOT NULL,
  predicted_price DOUBLE PRECISION NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  signal TEXT NOT NULL,
  horizon_minutes INTEGER NOT NULL,
  factor_json JSONB NOT NULL,
  evaluated_at TIMESTAMPTZ,
  realized_return DOUBLE PRECISION,
  is_correct BOOLEAN
);

CREATE TABLE IF NOT EXISTS factor_weights (
  factor_name TEXT PRIMARY KEY,
  weight DOUBLE PRECISION NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
