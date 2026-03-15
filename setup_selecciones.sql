-- PalomoFacts: Selecciones (National Teams) tables
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard → SQL Editor

-- National Team Researches
CREATE TABLE IF NOT EXISTS national_team_researches (
  id             UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title          TEXT NOT NULL DEFAULT 'Sin título',
  country        TEXT NOT NULL,
  confederation  TEXT DEFAULT '',
  config         JSONB DEFAULT '{}',
  results        JSONB DEFAULT '{}',
  created_at     TIMESTAMPTZ DEFAULT now(),
  updated_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_national_team_researches_updated ON national_team_researches (updated_at DESC);

ALTER TABLE national_team_researches ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all on national_team_researches" ON national_team_researches
  FOR ALL USING (true) WITH CHECK (true);


-- National Match Preps (between two national teams)
CREATE TABLE IF NOT EXISTS national_match_preps (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title         TEXT NOT NULL DEFAULT 'Sin título',
  home_country  TEXT NOT NULL,
  away_country  TEXT NOT NULL,
  tournament    TEXT DEFAULT '',
  config        JSONB DEFAULT '{}',
  results       JSONB DEFAULT '{}',
  created_at    TIMESTAMPTZ DEFAULT now(),
  updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_national_match_preps_updated ON national_match_preps (updated_at DESC);

ALTER TABLE national_match_preps ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all on national_match_preps" ON national_match_preps
  FOR ALL USING (true) WITH CHECK (true);


-- National Player Researches (international career focus)
CREATE TABLE IF NOT EXISTS national_player_researches (
  id           UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title        TEXT NOT NULL DEFAULT 'Sin título',
  player_name  TEXT NOT NULL,
  country      TEXT DEFAULT '',
  config       JSONB DEFAULT '{}',
  results      JSONB DEFAULT '{}',
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_national_player_researches_updated ON national_player_researches (updated_at DESC);

ALTER TABLE national_player_researches ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow all on national_player_researches" ON national_player_researches
  FOR ALL USING (true) WITH CHECK (true);
