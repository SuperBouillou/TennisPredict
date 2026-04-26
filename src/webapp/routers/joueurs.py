"""Router — Player profiles."""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.webapp.players import search_players, get_profile
from src.webapp.state import get_state
from src.webapp.utils import safe_float

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


# ── IOC country code → ISO-2 (for flag emoji) ─────────────────────────────────
_IOC_TO_ISO2 = {
    'ALG': 'DZ', 'ARG': 'AR', 'AUS': 'AU', 'AUT': 'AT', 'BEL': 'BE',
    'BIH': 'BA', 'BOL': 'BO', 'BRA': 'BR', 'BUL': 'BG', 'CAN': 'CA',
    'CHI': 'CL', 'CHN': 'CN', 'COL': 'CO', 'CRO': 'HR', 'CZE': 'CZ',
    'DEN': 'DK', 'ECU': 'EC', 'EGY': 'EG', 'ESP': 'ES', 'EST': 'EE',
    'FIN': 'FI', 'FRA': 'FR', 'GBR': 'GB', 'GEO': 'GE', 'GER': 'DE',
    'GRE': 'GR', 'HUN': 'HU', 'IND': 'IN', 'IRL': 'IE', 'ISR': 'IL',
    'ITA': 'IT', 'JPN': 'JP', 'KAZ': 'KZ', 'KOR': 'KR', 'LAT': 'LV',
    'LTU': 'LT', 'LUX': 'LU', 'MEX': 'MX', 'MNE': 'ME', 'MOR': 'MA',
    'NED': 'NL', 'NOR': 'NO', 'NZL': 'NZ', 'PAR': 'PY', 'PER': 'PE',
    'POL': 'PL', 'POR': 'PT', 'QAT': 'QA', 'ROU': 'RO', 'RSA': 'ZA',
    'RUS': 'RU', 'SER': 'RS', 'SLO': 'SI', 'SRB': 'RS', 'SUI': 'CH',
    'SVK': 'SK', 'SWE': 'SE', 'THA': 'TH', 'TPE': 'TW', 'TUN': 'TN',
    'TUR': 'TR', 'UAE': 'AE', 'UKR': 'UA', 'URU': 'UY', 'USA': 'US',
    'VEN': 'VE',
}

_IOC_TO_NAME = {
    'ALG': 'Algérie', 'ARG': 'Argentine', 'AUS': 'Australie', 'AUT': 'Autriche',
    'BEL': 'Belgique', 'BRA': 'Brésil', 'BUL': 'Bulgarie', 'CAN': 'Canada',
    'CHI': 'Chili', 'CHN': 'Chine', 'COL': 'Colombie', 'CRO': 'Croatie',
    'CZE': 'Rép. Tchèque', 'DEN': 'Danemark', 'ESP': 'Espagne', 'EST': 'Estonie',
    'FRA': 'France', 'GBR': 'Grande-Bretagne', 'GEO': 'Géorgie', 'GER': 'Allemagne',
    'GRE': 'Grèce', 'HUN': 'Hongrie', 'IND': 'Inde', 'IRL': 'Irlande',
    'ISR': 'Israël', 'ITA': 'Italie', 'JPN': 'Japon', 'KAZ': 'Kazakhstan',
    'KOR': 'Corée', 'LAT': 'Lettonie', 'LTU': 'Lituanie', 'MEX': 'Mexique',
    'MOR': 'Maroc', 'NED': 'Pays-Bas', 'NOR': 'Norvège', 'NZL': 'Nouvelle-Zélande',
    'POL': 'Pologne', 'POR': 'Portugal', 'QAT': 'Qatar', 'ROU': 'Roumanie',
    'RSA': 'Afrique du Sud', 'RUS': 'Russie', 'SER': 'Serbie', 'SRB': 'Serbie',
    'SLO': 'Slovénie', 'SUI': 'Suisse', 'SVK': 'Slovaquie', 'SWE': 'Suède',
    'THA': 'Thaïlande', 'TUN': 'Tunisie', 'TUR': 'Turquie', 'UKR': 'Ukraine',
    'USA': 'États-Unis',
}


def _flag_emoji(ioc: str) -> str:
    iso2 = _IOC_TO_ISO2.get((ioc or '').upper(), '')
    if len(iso2) != 2:
        return ''
    return chr(0x1F1E6 + ord(iso2[0]) - ord('A')) + chr(0x1F1E6 + ord(iso2[1]) - ord('A'))


def _age_from_dob(dob) -> int | None:
    if dob is None:
        return None
    try:
        if hasattr(dob, 'year'):       # datetime / Timestamp
            d = date(dob.year, dob.month, dob.day)
        elif isinstance(dob, str):
            d = date.fromisoformat(str(dob)[:10])
        else:
            return None
        today = date.today()
        return today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    except Exception:
        return None



def _best_surface(p: dict) -> str:
    return max(
        ('Hard', 'Clay', 'Grass'),
        key=lambda s: safe_float(p.get(f'elo_{s}'), 1500),
    )


def _build_radar(p: dict) -> dict:
    """8-axis radar data, values 0-100."""
    elo   = safe_float(p.get('elo'),       1500)
    elo_h = safe_float(p.get('elo_Hard'),  1500)
    elo_c = safe_float(p.get('elo_Clay'),  1500)
    elo_g = safe_float(p.get('elo_Grass'), 1500)

    def norm_elo(v: float) -> float:
        return round(max(0.0, min(100.0, (v - 1200) / 900 * 100)), 1)

    return {
        'labels': ['FORME', 'ELO', 'CLAY', 'HARD', 'GRASS', 'FITNESS', 'ENDURANCE', 'MENTAL'],
        'values': [
            round(safe_float(p.get('winrate_10'), 0.5) * 100, 1),
            norm_elo(elo),
            norm_elo(elo_c),
            norm_elo(elo_h),
            norm_elo(elo_g),
            round(max(0.0, 100.0 - safe_float(p.get('matches_14d'), 0) * 10), 1),
            round(safe_float(p.get('sets_ratio_10'), 0.5) * 100, 1),
            round(safe_float(p.get('tiebreak_winrate_10'), 0.5) * 100, 1),
        ],
    }


def _build_insights(p: dict) -> list[dict]:
    """Up to 4 dynamic insights derived from existing profile fields."""
    insights: list[dict] = []

    elo_surf = {s: safe_float(p.get(f'elo_{s}'), 1500) for s in ('Hard', 'Clay', 'Grass')}
    best_surf  = max(elo_surf, key=elo_surf.get)
    worst_surf = min(elo_surf, key=elo_surf.get)
    best_elo   = elo_surf[best_surf]
    worst_elo  = elo_surf[worst_surf]
    global_elo = safe_float(p.get('elo'), 1500)
    surf_adv   = best_elo - global_elo
    surf_gap   = best_elo - worst_elo

    # 1. Surface dominante
    if surf_adv > 50:
        insights.append({
            'ok': True,
            'title': f'Spécialiste {best_surf}',
            'sub':   f'ELO {best_surf} : {int(best_elo)} — +{int(surf_adv)} pts vs ELO global',
        })

    # 2. Forme récente
    streak = int(safe_float(p.get('streak'), 0))
    wr10   = safe_float(p.get('winrate_10'), 0.5)
    if streak >= 3:
        insights.append({
            'ok': True,
            'title': f'Série en cours ({streak} victoires)',
            'sub':   f'Win rate 10 derniers matchs : {int(wr10 * 100)}%',
        })
    elif wr10 >= 0.65:
        insights.append({
            'ok': True,
            'title': 'Forme excellente',
            'sub':   f'Win rate 10 matchs : {int(wr10 * 100)}%',
        })
    elif wr10 < 0.40:
        insights.append({
            'ok': False,
            'title': 'Forme en baisse',
            'sub':   f'Win rate 10 matchs seulement {int(wr10 * 100)}%',
        })

    # 3. Surface faiblesse
    if surf_gap > 120:
        insights.append({
            'ok': False,
            'title': f'Faible sur {worst_surf}',
            'sub':   f'ELO {worst_surf} : {int(worst_elo)} — {int(surf_gap)} pts sous {best_surf}',
        })

    # 4. Fatigue / rouille
    matches_14d = int(safe_float(p.get('matches_14d'), 0))
    days_since  = int(safe_float(p.get('days_since'), 0))
    if matches_14d >= 6:
        insights.append({
            'ok': False,
            'title': f'Fatigue détectée ({matches_14d} matchs / 14j)',
            'sub':   'Pénalité fatigue activée — impact négatif prédit',
        })
    elif days_since > 21 and matches_14d == 0:
        insights.append({
            'ok': None,
            'title': f'Absence récente ({days_since}j sans match)',
            'sub':   'Manque de compétition — incertitude accrue',
        })

    return insights[:4]


# ── Splits stub — structure ready for future implementation ────────────────────
#
# When splits are available in player_profiles_updated.parquet, replace
# _SPLITS_STUB with a function that reads p['split_bo3_wr'], p['split_bo5_wr'],
# p['split_top10_wr'], p['split_vs_lefty_wr'], p['split_tb_wr'], p['split_indoor_wr']
# and their corresponding record strings (e.g. p['split_bo3_rec'] = "62W-29L").
#
# add_splits_to_update_database() in update_database.py should compute:
#   - bo3_wr / bo5_wr: filter df_p by n_sets <= 3 or >= 4
#   - top10_wr: filter by opponent rank <= 10
#   - vs_lefty_wr: join with players identity (hand == 'L')
#   - tb_wr: tiebreak_winrate_10 already exists → use broader last-N window
#   - indoor_wr: filter df_p by surface == 'Carpet' or tourney_type == 'indoor'
#     (tennis-data 'court' column: 'Indoor' / 'Outdoor')

_SPLITS_LABELS = [
    ('BEST OF 3',    'split_bo3_wr',     'split_bo3_rec'),
    ('BEST OF 5',    'split_bo5_wr',     'split_bo5_rec'),
    ('TOP 10 ADV.',  'split_top10_wr',   'split_top10_rec'),
    ('VS GAUCHERS',  'split_lefty_wr',   'split_lefty_rec'),
    ('TIE-BREAKS',   'split_tb_wr',      'split_tb_rec'),
    ('INDOOR',       'split_indoor_wr',  'split_indoor_rec'),
]


def _build_splits(p: dict) -> list[dict] | None:
    """Returns splits data if available, else None (shows placeholder)."""
    available = [lbl for lbl, wr_key, _ in _SPLITS_LABELS if wr_key in p and p.get(wr_key) is not None]
    if not available:
        return None
    rows = []
    for lbl, wr_key, rec_key in _SPLITS_LABELS:
        wr  = safe_float(p.get(wr_key), 0.0)
        rec = p.get(rec_key, '')
        rows.append({'label': lbl, 'wr': round(wr * 100, 1), 'rec': rec or ''})
    return rows


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/joueurs", response_class=HTMLResponse)
async def joueurs_page(request: Request, tour: str = "atp"):
    return templates.TemplateResponse(request, "joueurs.html", {
        "active": "joueurs", "tour": tour,
    })


@router.get("/joueurs/search", response_class=HTMLResponse)
async def joueurs_search(request: Request, q: str = "", tour: str = "atp"):
    artifacts = get_state().get('models', {}).get(tour)
    if not artifacts or len(q) < 2:
        return HTMLResponse("")
    results = search_players(artifacts['profiles'], artifacts['players'], q)
    html = ""
    for p in results:
        name = p['player_name']
        rank = p.get('rank') or '—'
        elo  = int(p.get('elo', 0))
        surf = p.get('surface', '?')
        safe = name.replace("'", "\\'")
        html += (
            f'<div class="card" style="margin-bottom:6px;cursor:pointer" '
            f'onclick="window.location=\'/joueurs/{safe}?tour={tour}\'">'
            f'<strong>{name}</strong> '
            f'<span style="color:var(--muted);font-size:12px">'
            f'#{rank} · ELO {elo} · {surf}</span></div>'
        )
    return HTMLResponse(html or '<div style="color:var(--muted)">Aucun résultat.</div>')


@router.get("/joueurs/{player_name}/elo-history")
async def elo_history(player_name: str, tour: str = "atp"):
    """Return monthly ELO trajectory for last ~13 months. Used by Chart.js."""
    from src.config import get_paths

    paths = get_paths(tour)
    # Prefer the ELO-enriched match file; fall back to features file
    parquet_candidates = [
        paths['processed_dir'] / 'matches_with_elo.parquet',
        paths['processed_dir'] / 'matches_features_final.parquet',
    ]
    parquet_path = next((p for p in parquet_candidates if p.exists()), None)

    if parquet_path is None:
        return JSONResponse({"months": [], "elo": []})

    try:
        df = pd.read_parquet(
            parquet_path,
            columns=['p1_name', 'p2_name', 'p1_elo', 'p2_elo', 'tourney_date'],
        )
        name_lower = player_name.lower().strip()
        mask_p1 = df['p1_name'].str.lower() == name_lower
        mask_p2 = df['p2_name'].str.lower() == name_lower

        rows_p1 = df[mask_p1][['tourney_date', 'p1_elo']].rename(columns={'p1_elo': 'elo'})
        rows_p2 = df[mask_p2][['tourney_date', 'p2_elo']].rename(columns={'p2_elo': 'elo'})
        combined = pd.concat([rows_p1, rows_p2]).dropna().sort_values('tourney_date')

        if combined.empty:
            return JSONResponse({"months": [], "elo": []})

        # Last 13 months
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=13)
        combined = combined[combined['tourney_date'] >= cutoff]
        if combined.empty:
            return JSONResponse({"months": [], "elo": []})

        combined['month'] = pd.to_datetime(combined['tourney_date']).dt.to_period('M').astype(str)
        monthly = combined.groupby('month')['elo'].last().reset_index()

        return JSONResponse({
            "months": monthly['month'].tolist(),
            "elo":    [round(float(v), 0) for v in monthly['elo'].tolist()],
        })
    except Exception as exc:
        return JSONResponse({"months": [], "elo": [], "error": str(exc)})


@router.get("/joueurs/{player_name:path}", response_class=HTMLResponse)
async def joueur_profile(request: Request, player_name: str, tour: str = "atp"):
    artifacts = get_state().get('models', {}).get(tour)
    if not artifacts:
        return HTMLResponse("Circuit non disponible.", status_code=503)

    profile = get_profile(artifacts['profiles'], artifacts['players'], player_name)
    if not profile:
        return HTMLResponse(
            f'<div class="content"><div class="card" style="color:var(--red)">'
            f'Joueur "{player_name}" non trouvé.</div></div>',
            status_code=404,
        )

    ioc  = str(profile.get('ioc') or profile.get('country') or '').upper()
    hand = str(profile.get('hand') or '').upper()

    return templates.TemplateResponse(request, "joueurs_profile.html", {
        "active":      "joueurs",
        "tour":        tour,
        "p":           profile,
        # Pre-computed extras
        "flag":        _flag_emoji(ioc),
        "ioc":         ioc,
        "country_name": _IOC_TO_NAME.get(ioc, ioc),
        "hand_label":  {'R': 'Droitier', 'L': 'Gaucher', 'A': 'Ambidextre'}.get(hand, '—'),
        "age":         _age_from_dob(profile.get('dob')),
        "height":      profile.get('height'),
        "best_surf":   _best_surface(profile),
        "radar":       _build_radar(profile),
        "insights":    _build_insights(profile),
        "splits":      _build_splits(profile),   # None until update_database.py is extended
    })
