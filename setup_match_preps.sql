-- Match Preps table for storing match preparation research history
CREATE TABLE IF NOT EXISTS match_preps (
  id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title       TEXT NOT NULL DEFAULT 'Sin título',
  home_team   TEXT NOT NULL,
  away_team   TEXT NOT NULL,
  tournament  TEXT DEFAULT '',
  match_type  TEXT DEFAULT '',
  stadium     TEXT DEFAULT '',
  config      JSONB DEFAULT '{}',
  results     JSONB DEFAULT '{}',
  created_at  TIMESTAMPTZ DEFAULT now(),
  updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Index for fast ordering
CREATE INDEX IF NOT EXISTS idx_match_preps_updated ON match_preps (updated_at DESC);

-- Enable RLS
ALTER TABLE match_preps ENABLE ROW LEVEL SECURITY;

-- Allow all operations for anon key (single-user app)
CREATE POLICY "Allow all on match_preps" ON match_preps
  FOR ALL USING (true) WITH CHECK (true);
