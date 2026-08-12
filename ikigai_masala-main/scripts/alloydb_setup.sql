-- =============================================================================
-- IKIGAI MASALA — AlloyDB setup (database: menu_engineering)
-- =============================================================================
-- Postgres/AlloyDB DDL that reproduces the tool's CURRENT Supabase schema
-- EXACTLY — same 4 tables, same columns, same JSONB shapes — so the AlloyDB
-- migration changes the connection, not the data model. Idempotent: safe to
-- re-run (CREATE TABLE IF NOT EXISTS + ALTER ... ADD COLUMN IF NOT EXISTS).
--
-- Run in AlloyDB Studio against database: menu_engineering
--   (or: psql "$DATABASE_URL" -f scripts/alloydb_setup.sql)
--
-- This is the AlloyDB twin of scripts/setup_all.sql (the Supabase master).
-- It carries the final consolidated schema only; it does NOT fold the old
-- normalized config tables (that migration already ran on Supabase — the data
-- you import here is already in the counters JSONB shape).
-- =============================================================================

BEGIN;

-- 1. Clients — the whole per-client config lives in the counters JSONB column.
--    counters[0] is the primary counter the solver plans from; extra entries
--    are additional cuisine stations. Every other column is a plain per-client
--    attribute (city, weekend service, cooldown window, item pools, launch
--    flag, cross-counter shared categories).
CREATE TABLE IF NOT EXISTS clients (
    name               TEXT PRIMARY KEY,
    version            INT  NOT NULL DEFAULT 1,
    counters           JSONB NOT NULL DEFAULT '[]'::jsonb,
    city               TEXT,
    serve_weekends     BOOLEAN NOT NULL DEFAULT false,
    item_cooldown_days INT,
    source_pools       JSONB,
    working_days       JSONB,
    is_launch_site     BOOLEAN NOT NULL DEFAULT false,
    shared_categories  JSONB,
    created_at         TIMESTAMPTZ DEFAULT now()
);

-- 2. App-level settings (core_min_one_slots, constant_slots, fallback, etc.)
CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT  PRIMARY KEY,
    value JSONB NOT NULL
);

-- 3. Menu history — one row per (client, service_date); the day's whole menu
--    lives in the `menu` JSONB column ({slot: item_base}, or nested
--    {counter: {slot: item_base}} for multi-counter clients). Item-level
--    cooldowns explode this in memory.
CREATE TABLE IF NOT EXISTS menu_history (
    client_name  TEXT NOT NULL REFERENCES clients(name) ON DELETE CASCADE,
    service_date DATE NOT NULL,
    menu         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (client_name, service_date)
);

-- 4. Week signatures — one row per saved week plan (week-level cooldowns).
CREATE TABLE IF NOT EXISTS week_signatures (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    client_name     TEXT NOT NULL REFERENCES clients(name) ON DELETE CASCADE,
    week_start      DATE NOT NULL,
    week_signature  TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- Bring a partially-migrated database up to the full column set (idempotent).
-- Mirrors the ADD COLUMN IF NOT EXISTS block in scripts/setup_all.sql so an
-- AlloyDB instance seeded from an older dump gains the newer columns.
-- -----------------------------------------------------------------------------
ALTER TABLE clients ADD COLUMN IF NOT EXISTS version            INT   NOT NULL DEFAULT 1;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS counters           JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS city               TEXT;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS serve_weekends     BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS item_cooldown_days INT;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS source_pools       JSONB;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS working_days       JSONB;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS is_launch_site     BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE clients ADD COLUMN IF NOT EXISTS shared_categories  JSONB;

CREATE INDEX IF NOT EXISTS idx_menu_history_client_date
    ON menu_history(client_name, service_date DESC);
CREATE INDEX IF NOT EXISTS idx_week_signatures_client_date
    ON week_signatures(client_name, week_start DESC);

-- -----------------------------------------------------------------------------
-- Seed: app_settings (the same three keys the tool reads).
-- -----------------------------------------------------------------------------
INSERT INTO app_settings (key, value) VALUES
    ('core_min_one_slots', '["bread", "rice", "starter", "veg_dry", "welcome_drink", "curd_side", "nonveg_main", "veg_gravy"]'::jsonb),
    ('constant_slots', '["white_rice", "papad", "pickle", "chutney"]'::jsonb),
    ('fallback_menu_category', '"menu_cat_3"'::jsonb)
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;

COMMIT;

-- Sanity check
SELECT 'clients' AS tbl, count(*)::text AS n FROM clients
UNION ALL SELECT 'app_settings', count(*)::text FROM app_settings
UNION ALL SELECT 'menu_history', count(*)::text FROM menu_history
UNION ALL SELECT 'week_signatures', count(*)::text FROM week_signatures;
