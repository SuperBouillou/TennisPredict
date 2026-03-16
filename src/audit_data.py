# src/audit_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pathlib import Path

# Chemin absolu — fonctionne peu importe d'où on lance le script
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

print(f"📁 Lecture depuis : {PROCESSED_DIR}")  # Pour vérifier

def audit_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rapport de complétude colonne par colonne.
    """
    audit = pd.DataFrame({
        'dtype'        : df.dtypes,
        'missing_count': df.isnull().sum(),
        'missing_pct'  : (df.isnull().sum() / len(df) * 100).round(2),
        'unique_values': df.nunique(),
    }).sort_values('missing_pct', ascending=False)

    print("=" * 55)
    print("RAPPORT DE COMPLÉTUDE")
    print("=" * 55)
    print(audit[audit['missing_count'] > 0].to_string())
    return audit


def audit_distributions(df: pd.DataFrame) -> None:
    """
    Vérifie les distributions des variables clés.
    """
    print("\n" + "=" * 55)
    print("DISTRIBUTIONS CLÉS")
    print("=" * 55)

    # Surfaces
    print("\n📌 Surfaces :")
    print(df['surface'].value_counts(dropna=False))

    # Niveaux de tournoi
    print("\n📌 Niveaux (tourney_level) :")
    level_map = {'G': 'Grand Chelem', 'M': 'Masters 1000',
                 'A': 'ATP 250/500',  'D': 'Davis Cup', 'F': 'Finals'}
    print(df['tourney_level'].map(level_map).value_counts(dropna=False))

    # Best of
    print("\n📌 Format (best_of) :")
    print(df['best_of'].value_counts(dropna=False))

    # Rangs
    print("\n📌 Rangs (winner / loser) :")
    for col in ['winner_rank', 'loser_rank']:
        if col in df.columns:
            print(f"  {col}: min={df[col].min():.0f} | "
                  f"median={df[col].median():.0f} | "
                  f"max={df[col].max():.0f} | "
                  f"NaN={df[col].isnull().sum()}")


def audit_stats_coverage(df: pd.DataFrame) -> None:
    """
    Vérifie la couverture des stats de match (ace, df, etc.)
    Ces stats sont souvent manquantes pour les petits tournois.
    """
    stat_cols = [c for c in df.columns if c.startswith(('w_', 'l_'))]

    if not stat_cols:
        print("\n⚠️  Aucune colonne de stats trouvée.")
        return

    coverage = pd.DataFrame({
        'available_pct': (df[stat_cols].notna().sum() / len(df) * 100).round(1)
    }).sort_values('available_pct', ascending=False)

    print("\n" + "=" * 55)
    print("COUVERTURE DES STATS DE MATCH")
    print("=" * 55)
    print(coverage.to_string())

    # Couverture par niveau de tournoi
    print("\n📌 Couverture stats par niveau de tournoi :")
    key_col = 'w_ace'
    if key_col in df.columns:
        print(
            df.groupby('tourney_level')[key_col]
            .apply(lambda x: f"{x.notna().mean()*100:.1f}%")
            .to_string()
        )


def audit_temporal(df: pd.DataFrame) -> None:
    """
    Vérifie la cohérence temporelle et le volume par année.
    """
    print("\n" + "=" * 55)
    print("VOLUME PAR ANNÉE")
    print("=" * 55)
    print(df.groupby('year').size().rename('nb_matchs').to_string())


def plot_audit(df: pd.DataFrame) -> None:
    """
    Génère un rapport visuel rapide.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Audit des données ATP", fontsize=14, fontweight='bold')

    # 1. Volume par année
    df.groupby('year').size().plot(
        kind='bar', ax=axes[0, 0], color='steelblue'
    )
    axes[0, 0].set_title("Matchs par année")
    axes[0, 0].set_xlabel("")

    # 2. Répartition par surface
    df['surface'].value_counts().plot(
        kind='pie', ax=axes[0, 1], autopct='%1.1f%%'
    )
    axes[0, 1].set_title("Répartition par surface")
    axes[0, 1].set_ylabel("")

    # 3. Distribution des rangs (winner)
    if 'winner_rank' in df.columns:
        df['winner_rank'].clip(upper=500).dropna().plot(
            kind='hist', bins=50, ax=axes[1, 0], color='green', alpha=0.7
        )
        axes[1, 0].set_title("Distribution des rangs (vainqueurs)")
        axes[1, 0].set_xlabel("Rang ATP")

    # 4. Couverture stats par année
    if 'w_ace' in df.columns:
        coverage_by_year = (
            df.groupby('year')['w_ace']
            .apply(lambda x: x.notna().mean() * 100)
        )
        coverage_by_year.plot(
            kind='bar', ax=axes[1, 1], color='orange'
        )
        axes[1, 1].set_title("Couverture stats ace% par année")
        axes[1, 1].set_ylabel("% matchs avec stats")
        axes[1, 1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig("data/processed/audit_report.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n📊 Graphique sauvegardé : data/processed/audit_report.png")


if __name__ == "__main__":

    df = pd.read_parquet(PROCESSED_DIR / "matches_consolidated.parquet")
    print(f"🎾 Dataset chargé : {df.shape[0]:,} matchs, {df.shape[1]} colonnes\n")

    audit_completeness(df)
    audit_distributions(df)
    audit_stats_coverage(df)
    audit_temporal(df)
    plot_audit(df)