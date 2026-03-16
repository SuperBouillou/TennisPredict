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
