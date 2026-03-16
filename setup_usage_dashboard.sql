-- PalomoFacts: Usage ledger for admin cost dashboard
-- Run this in Supabase SQL Editor after the existing setup scripts.

CREATE TABLE IF NOT EXISTS usage_runs (
  id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  run_id              TEXT NOT NULL UNIQUE,
  source_type         TEXT NOT NULL,
  source_id           TEXT,
  workflow            TEXT NOT NULL,
  title               TEXT NOT NULL DEFAULT '',
  subject             TEXT NOT NULL DEFAULT '',
  totals              JSONB NOT NULL DEFAULT '{}'::jsonb,
  steps               JSONB NOT NULL DEFAULT '[]'::jsonb,
  estimated_cost_usd  DOUBLE PRECISION NOT NULL DEFAULT 0,
  ingest_source       TEXT NOT NULL DEFAULT 'runtime',
  created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_usage_runs_created_at ON usage_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_runs_workflow ON usage_runs (workflow, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_runs_source_type ON usage_runs (source_type, created_at DESC);

ALTER TABLE usage_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all on usage_runs" ON usage_runs
  FOR ALL USING (true) WITH CHECK (true);
