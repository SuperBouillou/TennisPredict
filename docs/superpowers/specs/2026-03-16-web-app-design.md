# Web App TennisPredict — Design Spec

**Date:** 2026-03-16
**Statut:** Draft
**Stack:** FastAPI + HTMX + Jinja2 + SQLite
**Remplacement:** Dashboard Streamlit existant (`src/dashboard.py`)

---

## 1. Objectif

Remplacer le dashboard Streamlit par une web app légère, rapide et installable sur mobile (PWA). L'app est utilisée quotidiennement pour :

1. Consulter les matchs du jour et leurs prédictions
2. Saisir manuellement un match pour obtenir une prédiction + edge
3. Enregistrer et résoudre des paris
4. Consulter les profils joueurs et les comparer
5. Analyser les performances du modèle (backtest, ROI, feature importance)

L'app tourne **localement** sur le même serveur que le pipeline ML. Elle n'est pas multi-utilisateur. Elle peut être déployée ultérieurement sur Railway/Render.

---

## 2. Architecture

### Stack technique

| Couche | Technologie | Rôle |
|---|---|---|
| Backend | FastAPI (Python) | API REST + serveur de pages HTML |
| Templates | Jinja2 | Rendu HTML côté serveur |
| Interactivité | HTMX | Swaps HTML sans JS framework |
| Styles | CSS vanilla (dark theme) | Aucun framework CSS externe |
| Base de données | SQLite (via `sqlite3` stdlib) | Paris, bankroll, settings |
| Modèles ML | Fichiers `.pkl` existants | Chargés une fois au démarrage |
| Graphiques | Chart.js (CDN) | Equity curve, barres ROI |
| PWA | manifest.json + service worker | Installation mobile |

### Pourquoi ce stack

- **Zéro dépendance JS** côté développement — HTMX = simple `<script>` tag
- **FastAPI** : déjà familier dans l'écosystème Python, async natif, OpenAPI auto-générée
- **SQLite** : remplace les fichiers `bets_atp.csv` / `bankroll_atp.json` dispersés — une seule base de données robuste
- **Chart.js** via CDN : courbe de bankroll sans build step

### Démarrage

```bash
uvicorn src.webapp.main:app --reload --port 8000
```

L'app est accessible sur `http://localhost:8000`.

---

## 3. Structure des fichiers

```
src/webapp/
├── main.py                  # Point d'entrée FastAPI, lifespan (chargement modèles)
├── db.py                    # SQLite — init schema, helpers CRUD
├── ml.py                    # Wrapper modèles ML (predict, edge, Kelly)
├── espn.py                  # Client ESPN (réutilise src/espn_client.py)
├── players.py               # Recherche joueurs, profils, comparaison
├── routers/
│   ├── today.py             # GET /today + HTMX partials
│   ├── predictions.py       # POST /predict + formulaire
│   ├── history.py           # GET/POST /history, /bets
│   ├── joueurs.py           # GET /joueurs, /joueurs/{id}
│   └── stats.py             # GET /stats + partials graphiques
├── templates/
│   ├── base.html            # Layout commun (sidebar, head, PWA)
│   ├── today.html
│   ├── predictions.html
│   ├── history.html
│   ├── joueurs.html
│   ├── joueurs_profile.html
│   ├── stats.html
│   └── partials/
│       ├── match_card.html
│       ├── bet_row.html
│       ├── player_card.html
│       └── prediction_result.html
├── static/
│   ├── app.css
│   ├── manifest.json        # PWA
│   └── sw.js                # Service worker (cache statique uniquement)
└── migrations/
    └── 001_init.sql         # Schéma initial SQLite
```

### Réutilisation du code existant

Le webapp **importe directement** les modules existants :
- `src.espn_client` → récupération matchs ESPN
- `src.config` → constantes, chemins, `get_paths(tour)`
- `src.update_database.build_name_mapping` → normalisation noms joueurs

Les fichiers `.pkl` (modèle, imputer, platt_scaler, feature_list) sont chargés **une seule fois** au démarrage FastAPI via `lifespan`.

---

## 4. Base de données SQLite

Remplace `bets_atp.csv`, `bets_wta.csv`, `bankroll_atp.json`, `bankroll_wta.json`.

### Schéma

```sql
-- Paris
CREATE TABLE bets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tour        TEXT NOT NULL,          -- 'atp' | 'wta'
    created_at  TEXT NOT NULL,          -- ISO 8601
    tournament  TEXT NOT NULL,
    surface     TEXT NOT NULL,          -- 'Hard' | 'Clay' | 'Grass'
    round       TEXT,
    p1_name     TEXT NOT NULL,
    p2_name     TEXT NOT NULL,
    bet_on      TEXT NOT NULL,          -- 'p1' | 'p2'
    prob        REAL NOT NULL,          -- probabilité modèle
    edge        REAL,                   -- edge calculé
    odd         REAL NOT NULL,          -- cote bookmaker
    stake       REAL NOT NULL,          -- mise en €
    kelly_frac  REAL,                   -- fraction Kelly recommandée
    status      TEXT DEFAULT 'pending', -- 'pending' | 'won' | 'lost'
    pnl         REAL DEFAULT 0,         -- P&L réalisé
    resolved_at TEXT                    -- ISO 8601, nullable
);

-- Bankroll (une ligne par circuit)
CREATE TABLE bankroll (
    tour        TEXT PRIMARY KEY,       -- 'atp' | 'wta'
    amount      REAL NOT NULL DEFAULT 1000.0,
    updated_at  TEXT NOT NULL
);

-- Settings
CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Valeurs initiales : min_edge=0.03, min_prob=0.55, kelly_fraction=0.25
```

### Migration depuis les fichiers CSV existants

Un script `src/webapp/migrate_csv.py` lit les fichiers `bets_atp.csv` / `bets_wta.csv` existants et les insère dans SQLite à la première exécution. Les fichiers CSV sont conservés en lecture seule (`.bak`).

---

## 5. API — Endpoints principaux

### Pages complètes (navigation sidebar)

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Redirige vers `/today` |
| GET | `/today` | Page matchs du jour |
| GET | `/predictions` | Page prédiction manuelle |
| GET | `/history` | Page historique paris |
| GET | `/joueurs` | Page profils joueurs |
| GET | `/stats` | Page statistiques backtest |

### Partials HTMX

| Méthode | Route | Déclencheur |
|---|---|---|
| GET | `/today/matches?tour=atp&date=2026-03-16` | Toggle ATP/WTA, date picker |
| POST | `/predictions/run` | Formulaire prédiction |
| GET | `/predictions/autocomplete?q=sinner&tour=atp` | Recherche joueur live |
| POST | `/bets` | Enregistrer un pari |
| POST | `/bets/{id}/resolve` | Résoudre un pari (gagné/perdu) |
| GET | `/joueurs/search?q=alcaraz&tour=atp` | Recherche joueur |
| GET | `/joueurs/{player_id}` | Profil complet joueur |
| GET | `/joueurs/compare?p1={id}&p2={id}` | Comparaison deux joueurs |
| GET | `/stats/equity?tour=atp&strategy=Kelly` | Données equity curve (Flat/Kelly/Percent) |
| GET | `/stats/roi-bookmakers?tour=atp` | ROI par bookmaker (Bet365/Pinnacle/Max/Avg) |
| GET | `/stats/roi-surface?tour=atp` | ROI par surface + niveau |
| GET | `/stats/features?tour=atp` | Top 15 feature importances |
| GET | `/stats/calibration?tour=atp` | Données calibration curve |
| POST | `/sync?tour=atp` | Déclenche mise à jour ESPN en arrière-plan |
| GET | `/sync/status?tour=atp` | État de la sync en cours (pour polling HTMX toutes les 3s) |

---

## 6. Pages — Détail fonctionnel

### 6.1 Matchs du jour (`/today`)

**Source de données :** ESPN via `espn_client.py` — mis en cache 5 min (TTL).

**Fonctionnalités :**
- Toggle ATP / WTA (HTMX swap `#match-list`)
- Date picker ← J-1 / Aujourd'hui / J+1 → (HTMX swap)
- Pour chaque match : prédiction ML automatique si les deux joueurs sont dans la base
- Tri : value bets en premier (edge > seuil), puis par probabilité décroissante
- Card : badge VALUE BET / EDGE MODÉRÉ / neutre, nom des joueurs, probabilité, Kelly en % et en €, bouton "Parier →"
- Clic "Parier →" : ouvre un modal inline (HTMX) pour saisir la mise et confirmer
- Barre de bankroll en bas de page (live depuis SQLite)
- Badge "Dernière sync il y a Xmin" + bouton ↻

**Gestion d'erreur :** Si ESPN est inaccessible → message "Données ESPN indisponibles — saisie manuelle dans Prédictions".

### 6.2 Prédictions manuelles (`/predictions`)

**Fonctionnalités :**
- Toggle ATP / WTA (change le pool de joueurs pour l'autocomplete)
- **Autocomplete Joueur 1 / Joueur 2** : frappe 2+ caractères → HTMX GET `/predictions/autocomplete` → liste déroulante des joueurs correspondants (nom + classement actuel)
- Sélection tournoi (dropdown pré-rempli depuis `matches_features_final.parquet`)
- Surface (Hard / Clay / Grass)
- Tour (R128 → Finale)
- Best-of (3 ou 5)
- Cotes optionnelles : Joueur 1 / Joueur 2 (deux inputs numériques)
- Bouton ⚡ Analyser → POST HTMX → remplace `#result-card`

**Résultat affiché :**
- Probabilité ML (calibrée, bidirectionnelle)
- Probabilité ELO (baseline)
- Edge (si cotes saisies) + couleur verte/orange/rouge
- EV (expected value)
- Kelly recommandé (fraction × bankroll = X €)
- Bouton "Enregistrer le pari" → POST `/bets`

**Gestion d'erreur :** Joueur introuvable → warning inline "Joueur non trouvé — prédiction basée sur ELO uniquement".

### 6.3 Historique des paris (`/history`)

**Fonctionnalités :**
- En-tête : bankroll actuelle ATP + WTA, mise totale en jeu, P&L du mois
- **Exposition du jour** : bandeau si paris en attente aujourd'hui (X€ risqués, gain potentiel Y€)
- Section "En attente" : boutons Gagné / Perdu → POST `/bets/{id}/resolve` → swap HTMX
- Section "Résolus" : tableau avec date, match, cote, mise, P&L
- **Filtres** (HTMX) : circuit (ATP/WTA), surface, mois, statut
- **Pagination** : 20 paris par page
- Bouton Export CSV (GET `/history/export`)

**Calcul bankroll :** Pessimiste — la mise est débitée immédiatement à l'enregistrement (`bankroll -= stake`). Sur résolution "Gagné" : `bankroll += stake × (odd - 1)` (profit net, la mise étant déjà déduite). Sur "Perdu" : bankroll reste inchangée (déjà débitée). Exemple : mise=100€, cote=2.0 → au pari `bankroll -= 100`, sur victoire `bankroll += 100`. Gain net : 0€ (break-even à cote 1.0) ou `stake × (odd-1)` pour toute autre cote.

### 6.4 Profils Joueurs (`/joueurs`)

**Vue liste :**
- Champ de recherche → HTMX GET `/joueurs/search` → cards joueurs
- Chaque card : nom, classement, ELO, win rate surface principale
- **Comparateur** : sélectionner deux joueurs → bouton Comparer → GET `/joueurs/compare`

**Clé `player_id` :** entier Sackmann issu de `players.parquet` (colonne `player_id`). L'autocomplete retourne `{name, player_id, rank}`. Les routes `/joueurs/{player_id}` et `/joueurs/compare?p1=X&p2=Y` utilisent cet entier.

**Jointure entre les deux sources :** `player_profiles_updated.parquet` identifie les joueurs via `player_name` (ex : `"Carlos Alcaraz"`). `players.parquet` a `name_first` + `name_last`. La clé de jointure utilisée dans tout le codebase existant (`update_database.py` ligne 363, `predict_today.py` lignes 59/90) est :

```python
name_key = full_name.lower().strip()  # ex : "carlos alcaraz"
```

Dans `ml.py`, la jointure est donc : `players['name_key'] = (players['name_first'] + ' ' + players['name_last']).str.lower().str.strip()` — à joindre sur `player_profiles_updated['player_name'].str.lower().str.strip()`.

**Vue profil (`/joueurs/{player_id}`) :**
- Identité : nom, nationalité (`ioc`), main (`hand`), âge (calculé depuis `dob`), classement ATP
- Sources : `players.parquet` (identité) + `player_profiles_updated.parquet` (stats récentes) — jointes sur `name_key`
- ELO global + Hard / Clay / Grass (barres comparées à la moyenne du tour)
- Glicko RD (incertitude sur le rating) — affiché comme "confiance dans le rating"
- Forme récente : win rates 5 / 10 / 20 matchs (mini sparkline Chart.js)
- **Graphique ELO dans le temps** (Chart.js ligne, 2 ans glissants)
- Surface de prédilection (meilleur win rate)
- H2H vs top 10 (win rate agrégé)
- Derniers matchs (tableau 10 lignes)

**Vue comparaison (`/joueurs/compare?p1=X&p2=Y`) :**
- Deux colonnes côte à côte : tous les indicateurs en miroir
- ELO / classement / win rates / surface / H2H direct entre eux
- Tableau H2H détaillé (tous les matchs joués ensemble, surface, résultat)

### 6.5 Statistiques (`/stats`)

**KPIs :** ROI, Sharpe, nombre de paris, win rate — par circuit (toggle ATP/WTA).

**Graphique 1 — Equity curve (`GET /stats/equity?tour=atp&strategy=Kelly`) :**
- Source : `backtest_kelly.parquet` (ou `flat` / `percent` selon le paramètre `strategy`) — fichiers générés par `src/backtest.py` dans `data/models/{tour}/`
- Réponse JSON : `{ labels: ["2025-01-05", ...], values: [1000, 1003, ...] }`
- Toggle stratégie (Flat / Kelly / Percent) → HTMX swap du canvas Chart.js
- Axe X : date (semaines), Axe Y : bankroll en €, baseline 1 000€ en pointillés
- Tooltip au survol : date, bankroll, P&L semaine

**Graphique 2 — ROI par bookmaker (`GET /stats/roi-bookmakers?tour=atp`) :**
- Source : `backtest_real_{Bookmaker}.parquet` pour Bet365, Pinnacle, Max, Avg
- Réponse JSON : `{ bookmakers: ["Bet365", "Pinnacle", "Max", "Avg"], roi: [0.02, 0.055, 0.07, 0.04] }`
- Barres groupées, couleur positive (vert) / négative (rouge)

**ROI par surface et niveau :** barres horizontales (existant dans la maquette).

**Feature importance (Chart.js barres horizontales) :**
- Top 15 features par importance XGBoost
- Colorées par groupe : ELO (bleu) / Forme (vert) / H2H (orange) / Stats service (violet)

**Calibration curve :**
- Courbe de calibration du modèle (probabilité prédite vs fréquence réelle)
- Droite de calibration parfaite en pointillés

**Configuration des seuils (section bas de page) :**
- Sliders : edge minimum (0–10%), probabilité minimum (50–70%), Kelly fraction (0.1–0.5)
- Sauvegarde → POST `/settings` → mise à jour **SQLite uniquement** (table `settings`)
- **SQLite est la source de vérité pour les seuils en runtime.** `optimal_thresholds.json` est traité comme un fichier en lecture seule généré par `optimize_thresholds.py` — le webapp ne l'écrit jamais. Au démarrage, si la table `settings` est vide, les valeurs sont initialisées depuis `optimal_thresholds.json` puis restent dans SQLite.

---

## 7. Chargement des modèles ML

Au démarrage FastAPI (via `lifespan`), chargement depuis `data/models/{tour}/` :

> **Note :** `get_paths(tour)` retourne des clés de **répertoires** (`models_dir`, `processed_dir`, etc.), pas des chemins de fichiers. Les chemins sont construits par concaténation.

```python
models = {}
for tour in ['atp', 'wta']:
    paths = get_paths(tour)
    models_dir    = paths['models_dir']
    processed_dir = paths['processed_dir']
    models[tour] = {
        'model':        joblib.load(models_dir / 'xgb_tuned.pkl'),
        'imputer':      joblib.load(models_dir / 'imputer.pkl'),
        'platt':        joblib.load(models_dir / 'platt_scaler.pkl'),
        'feature_list': joblib.load(models_dir / 'feature_list.pkl'),
        'profiles':     pd.read_parquet(processed_dir / 'player_profiles_updated.parquet'),
        'elo':          pd.read_parquet(processed_dir / 'elo_ratings_updated.parquet'),
        'players':      pd.read_parquet(processed_dir / 'players.parquet'),  # Sackmann — nationalité, main, DOB, player_id
    }
```

La fonction `ml.predict(tour, p1_name, p2_name, tournament, surface, round, best_of, odd_p1, odd_p2)` retourne un dict :

```python
{
  'prob_p1': 0.672,
  'prob_p2': 0.328,
  'elo_prob': 0.641,
  'edge': 0.052,        # None si pas de cotes
  'ev': 0.041,          # None si pas de cotes
  'kelly_frac': 0.021,  # None si pas de cotes
  'kelly_eur': 26.2,    # None si pas de cotes
  'p1_found': True,
  'p2_found': True,
  'confidence': 0.672
}
```

---

## 8. PWA

**`manifest.json` :**
```json
{
  "name": "TennisPredict",
  "short_name": "Tennis",
  "start_url": "/today",
  "display": "standalone",
  "background_color": "#0f172a",
  "theme_color": "#3b82f6",
  "icons": [{ "src": "/static/icon-192.png", "sizes": "192x192" }]
}
```

**Service worker (`sw.js`) :** cache uniquement les assets statiques (CSS, Chart.js, icônes). Les pages HTML ne sont pas mises en cache (données live).

---

## 9. Sync automatique

> **Note :** `scheduled_sync.py` est un wrapper CLI pour Windows Task Scheduler — ne pas l'appeler directement depuis FastAPI. Le bon point d'entrée est `fetch_live_data.run_update(tour, force=False)`.

La sync est déclenchée via `asyncio.to_thread(fetch_live_data.run_update, tour)` dans un `BackgroundTask` FastAPI, dans deux cas :
- Au démarrage de l'app (si dernière sync > 6h, vérifié via `settings` SQLite)
- Sur clic bouton ↻ dans l'interface (POST `/sync?tour=atp`)

La sync met à jour : ESPN matches → player profiles → ELO ratings → predictions CSV. Durée typique : 30–90s selon le circuit. Un indicateur "Sync en cours…" est affiché via polling HTMX (`hx-get="/sync/status"` toutes les 3s).

---

## 10. Gestion des erreurs

| Situation | Comportement |
|---|---|
| ESPN inaccessible | Message inline, formulaire manuel toujours disponible |
| Joueur introuvable | Warning inline, prédiction ELO-only si ELO disponible |
| Modèle non chargé | 503 avec message clair |
| Paris sur match déjà résolu | Validation côté serveur, erreur 400 inline |
| SQLite corrompu | Backup automatique `.bak` au démarrage |

---

## 11. Tests

- `tests/webapp/test_ml.py` — tests unitaires `ml.predict()` avec fixtures (joueur connu, inconnu, ELO-only fallback)
- `tests/webapp/test_db.py` — tests CRUD bets + bankroll : enregistrement, résolution Gagné/Perdu, vérification formule bankroll (`stake × (odd-1)`), migration CSV
- `tests/webapp/test_routes.py` — tests d'intégration FastAPI (TestClient) :
  - Toutes les routes de pages retournent 200 avec `Content-Type: text/html`
  - **Les partials HTMX retournent un fragment HTML** (pas une page complète — vérifier l'absence de `<html>` dans la réponse)
  - Chemin ESPN inaccessible → 200 avec message de fallback (pas 500)
  - Chemin joueur introuvable → prédiction ELO-only + warning inline
  - POST `/bets/{id}/resolve` sur pari déjà résolu → 400
- `tests/webapp/test_autocomplete.py` — recherche joueurs : accents (`Müller`), noms composés (`Bautista Agut`), casse indifférente, minimum 2 caractères, résultats triés par classement

---

## 12. Hors scope v1

- Authentification / multi-utilisateur
- Intégration API bookmaker (Pinnacle API, Betfair Exchange)
- Suivi du mouvement des cotes (odds drift)
- Notifications push (Web Push API)
- Gestion parlays / combinés
- Déploiement public (Railway/Render) — prévu v2
