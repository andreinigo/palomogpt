-- PalomoFacts: Team & Player Research tables
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard → SQL Editor

-- Team Researches
CREATE TABLE IF NOT EXISTS team_researches (
  id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title       TEXT NOT NULL DEFAULT 'Sin título',
  team_name   TEXT NOT NULL,
  tournament  TEXT DEFAULT '',
  config      JSONB DEFAULT '{}',
  results     JSONB DEFAULT '{}',
  created_at  TIMESTAMPTZ DEFAULT now(),
  updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_team_researches_updated ON team_researches (updated_at DESC);

ALTER TABLE team_researches ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all on team_researches" ON team_researches
  FOR ALL USING (true) WITH CHECK (true);


-- Player Researches
CREATE TABLE IF NOT EXISTS player_researches (
  id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title        TEXT NOT NULL DEFAULT 'Sin título',
  player_name  TEXT NOT NULL,
  team_name    TEXT DEFAULT '',
  position     TEXT DEFAULT '',
  config       JSONB DEFAULT '{}',
  results      JSONB DEFAULT '{}',
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_player_researches_updated ON player_researches (updated_at DESC);

ALTER TABLE player_researches ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all on player_researches" ON player_researches
  FOR ALL USING (true) WITH CHECK (true);
