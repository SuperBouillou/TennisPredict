CREATE TABLE IF NOT EXISTS bets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tour        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    tournament  TEXT NOT NULL,
    surface     TEXT NOT NULL,
    round       TEXT,
    p1_name     TEXT NOT NULL,
    p2_name     TEXT NOT NULL,
    bet_on      TEXT NOT NULL,
    prob        REAL NOT NULL,
    edge        REAL,
    odd         REAL NOT NULL,
    stake       REAL NOT NULL,
    kelly_frac  REAL,
    status      TEXT DEFAULT 'pending',
    pnl         REAL DEFAULT 0,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS bankroll (
    tour       TEXT PRIMARY KEY,
    amount     REAL NOT NULL DEFAULT 1000.0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT DEFAULT (datetime('now'))
);

-- Track record automatique : un signal VALUE = une ligne insérée automatiquement
-- Résolution automatique chaque nuit via le cron daily_update.sh
CREATE TABLE IF NOT EXISTS signal_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,          -- datetime ISO UTC du signal
    tour         TEXT NOT NULL,          -- 'atp' | 'wta'
    tournament   TEXT,
    surface      TEXT,
    level        TEXT,
    round        TEXT,
    p1_name      TEXT NOT NULL,
    p2_name      TEXT NOT NULL,
    bet_on       TEXT NOT NULL,          -- joueur sur lequel parier
    prob_model   REAL,                   -- probabilité modèle pour bet_on
    odd_snapshot REAL,                   -- cote Pinnacle capturée au signal
    edge         REAL,                   -- edge au moment du signal
    stake_units  REAL DEFAULT 1.0,       -- toujours 1 unité flat
    result       TEXT DEFAULT 'pending', -- 'pending' | 'won' | 'lost' | 'void'
    pnl_units    REAL,                   -- profit/perte en unités (null si pending)
    resolved_at  TEXT                    -- datetime ISO UTC de la résolution
);
