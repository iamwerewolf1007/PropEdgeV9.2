"""
PropEdge V9.2 — Projection Model Trainer
==========================================
- Live rolling stats via vectorised pandas (never reads L*_* CSV columns)
- All rolling cols built with pd.concat — no PerformanceWarning
- GBR uses early stopping — no hang on large datasets
- DNP rows excluded from training data
- Matches old training row count: min 10 prior played games per row
"""
import pandas as pd
import numpy as np
import pickle, json
from sklearn.ensemble import GradientBoostingRegressor
from config import get_dvp, POS_MAP

FEATURES = [
    'l30','l10','l5','l3','volume','trend','std10','defP','pace_rank',
    'h2h_ts_dev','h2h_fga_dev','h2h_min_dev','h2h_conf',
    'min_cv','pts_per_min','recent_min_trend','fga_per_min',
    'is_b2b','rest_days','consistency','line'
]


def build_training_data(file_2425, file_2526, file_h2h):
    """
    Build training samples using vectorised rolling.
    DNP rows excluded before rolling. Min 10 prior played games per row.
    shift(1) ensures each row only sees prior games — no lookahead.
    All rolling cols built via pd.concat — no PerformanceWarning.
    """
    from rolling_engine import filter_played

    df25 = pd.read_csv(file_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(file_2526, parse_dates=['GAME_DATE'])
    h2h  = pd.read_csv(file_h2h)

    # Datetime sort across both seasons
    # Add DNP column BEFORE concat so the resulting frame is never fragmented
    if 'DNP' not in df25.columns: df25 = df25.copy(); df25['DNP'] = 0
    if 'DNP' not in df26.columns: df26 = df26.copy(); df26['DNP'] = 0
    combined = pd.concat([df25, df26], ignore_index=True)
    combined = combined.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)

    # Exclude DNP rows from training entirely
    combined = filter_played(combined).copy()
    combined = combined.reset_index(drop=True)

    h2h_dedup = h2h.drop_duplicates(subset=['PLAYER_NAME','OPPONENT'], keep='last')
    h2h_lkp = {(r['PLAYER_NAME'], r['OPPONENT']): r.to_dict()
               for _, r in h2h_dedup.iterrows()}
    print(f"    Rows (played only): {len(combined):,}  Players: {combined['PLAYER_NAME'].nunique():,}")

    # ── Vectorised rolling — all cols built at once via pd.concat ──────────────
    grp = combined.groupby('PLAYER_NAME', sort=False)

    def sroll(col, w):
        """Shift(1) rolling mean — each row sees only prior games."""
        return grp[col].transform(
            lambda s: s.rolling(w, min_periods=1).mean().shift(1)
        )

    rolled = pd.concat([
        sroll('PTS', 30).rename('_l30'),
        sroll('PTS', 10).rename('_l10'),
        sroll('PTS',  5).rename('_l5'),
        sroll('PTS',  3).rename('_l3'),
        grp['PTS'].transform(
            lambda s: s.rolling(10, min_periods=3).std().shift(1)
        ).fillna(5.0).rename('_std10'),
        sroll('MIN_NUM', 10).rename('_m10'),
        sroll('MIN_NUM',  3).rename('_m3'),
        sroll('FGA',     10).rename('_fga10'),
        grp['GAME_DATE'].transform(
            lambda s: s.diff().dt.days.fillna(99)
        ).astype(int).rename('_rest'),
        grp['GAME_DATE'].transform('cumcount').rename('_seq'),
    ], axis=1)

    # Single concat — no fragmented inserts — no PerformanceWarning
    base = pd.concat([
        combined[['PLAYER_NAME', 'GAME_DATE', 'PTS', 'OPPONENT', 'PLAYER_POSITION']],
        rolled
    ], axis=1)

    # Require ≥10 prior played games (seq=10 → 11th game, same as old range(10,len))
    base = base[base['_seq'] >= 10].dropna(subset=['_l30']).copy()
    print(f"    Training rows after filters: {len(base):,}")

    # ── Vectorised feature computation ────────────────────────────────────────
    base['_l30']   = base['_l30'].astype(float)
    base['_l10']   = base['_l10'].astype(float)
    base['_l5']    = base['_l5'].astype(float)
    base['_l3']    = base['_l3'].astype(float)
    base['_std10'] = base['_std10'].astype(float)
    base['_m10']   = base['_m10'].fillna(28.0).astype(float)
    base['_m3']    = base['_m3'].fillna(28.0).astype(float)
    base['_fga10'] = base['_fga10'].fillna(8.0).astype(float)
    base['_rest']  = base['_rest'].astype(float)

    base['line']             = (base['_l30'] * 2).round() / 2
    base['volume']           = (base['_l30'] - base['line']).round(1)
    base['trend']            = (base['_l5']  - base['_l30']).round(1)
    m10c                     = base['_m10'].clip(lower=1)
    base['min_cv']           = (base['_std10'] / m10c).round(3)
    base['pts_per_min']      = (base['_l10'] / m10c).round(3)
    base['recent_min_trend'] = (base['_m3'] - base['_m10']).round(1)
    base['fga_per_min']      = (base['_fga10'] / m10c).round(3)
    base['is_b2b']           = (base['_rest'] == 1).astype(int)
    base['rest_days']        = base['_rest'].astype(int)
    base['consistency']      = (1 / (base['_std10'] + 1)).round(3)

    # pace_rank: real per-row value from opponent team FGA (same method as batch_predict)
    # FIX: was hardcoded=15 so GBR had zero variance → could not learn pace signal
    team_fga_mean = combined.groupby('OPPONENT')['FGA'].mean()
    fga_sorted    = team_fga_mean.sort_values(ascending=False)
    pace_rank_map = {team: i + 1 for i, team in enumerate(fga_sorted.index)}
    base['pace_rank'] = base['OPPONENT'].map(pace_rank_map).fillna(15).astype(int)

    # DVP per row
    def _dvp(row):
        pos = POS_MAP.get(str(row['PLAYER_POSITION']), 'Forward')
        return get_dvp(row['OPPONENT'], pos)
    base['defP'] = base.apply(_dvp, axis=1)

    # H2H per row
    def _h2h(row):
        hr = h2h_lkp.get((row['PLAYER_NAME'], row['OPPONENT']))
        if hr is None: return 0.0, 0.0, 0.0, 0.0
        def safe(k): return float(hr[k]) if pd.notna(hr.get(k)) else 0.0
        return safe('H2H_TS_VS_OVERALL'), safe('H2H_FGA_VS_OVERALL'), \
               safe('H2H_MIN_VS_OVERALL'), safe('H2H_CONFIDENCE')
    h2h_vals = base.apply(_h2h, axis=1, result_type='expand')
    h2h_vals.columns = ['h2h_ts_dev','h2h_fga_dev','h2h_min_dev','h2h_conf']
    base = pd.concat([base, h2h_vals], axis=1)

    # Rename for FEATURES list
    base = base.rename(columns={'_l30':'l30','_l10':'l10','_l5':'l5','_l3':'l3','_std10':'std10'})
    base['actual_pts'] = combined.loc[base.index, 'PTS'].astype(int)
    print(f"    Training samples: {len(base):,}")
    return base


def train_and_save(file_2425, file_2526, file_h2h, model_file, trust_file):
    """Train GBR with early stopping. Save model + player trust scores."""
    print("    Building training data (vectorised, DNP excluded)...")
    train_df = build_training_data(file_2425, file_2526, file_h2h)

    X = train_df[FEATURES].fillna(0)
    y = train_df['actual_pts']

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        n_iter_no_change=15,   # early stopping — prevents hang
        validation_fraction=0.1,
        tol=1e-4,
        random_state=42,
    )
    model.fit(X, y)
    print(f"    GBR: {model.n_estimators_} trees used (early stop from max 300)")

    model_file.parent.mkdir(parents=True, exist_ok=True)
    with open(model_file, 'wb') as f: pickle.dump(model, f)
    print(f"    ✓ Model → {model_file.name}")

    train_df['pred'] = model.predict(X)
    train_df['correct'] = (
        ((train_df['pred'] > train_df['line']) & (train_df['actual_pts'] > train_df['line'])) |
        ((train_df['pred'] < train_df['line']) & (train_df['actual_pts'] < train_df['line']))
    ).astype(int)

    trust = {
        p: round(float(g['correct'].mean()), 3)
        for p, g in train_df.groupby('PLAYER_NAME')
        if len(g) >= 10
    }
    with open(trust_file, 'w') as f: json.dump(trust, f, indent=2)
    print(f"    ✓ Trust: {len(trust)} players → {trust_file.name}")
    print(f"    In-sample accuracy: {float(train_df['correct'].mean()):.1%}")
    return model
