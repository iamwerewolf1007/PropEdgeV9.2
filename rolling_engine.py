"""
PropEdge V9.2 — Rolling Stats Engine
======================================
Live rolling stat computation. Never reads pre-computed L*_* CSV columns.
Career-chronological across both seasons. DNP rows excluded from all windows.

DNP definition: DNP=1 OR MIN_NUM=0 OR MIN_NUM is NaN
DNP rows are preserved in the CSV for audit but never enter rolling windows.
"""
import pandas as pd
import numpy as np
from config import WINDOWS, ROLL_COLS


# ─── DNP UTILITIES ───────────────────────────────────────────
def is_dnp_row(row):
    """Return True if this game row is a DNP."""
    if row.get('DNP', 0) == 1:
        return True
    mn = row.get('MIN_NUM', None)
    if mn is None or (isinstance(mn, float) and np.isnan(mn)):
        return True
    return float(mn) == 0.0


def filter_played(df):
    """
    Return only rows where the player actually played.
    Excludes DNP=1, MIN_NUM=0, MIN_NUM=NaN.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    mask = pd.Series(True, index=df.index)
    if 'DNP' in df.columns:
        mask = mask & (df['DNP'].fillna(0) != 1)
    if 'MIN_NUM' in df.columns:
        mask = mask & (df['MIN_NUM'].fillna(0) > 0)
    return df[mask].copy()


# ─── DATA LOADING ─────────────────────────────────────────────
def load_combined(file_2425, file_2526):
    """
    Load both season CSVs, combine, datetime-sort by player+date.
    Returns combined DataFrame with DNP column ensured.
    """
    df25 = pd.read_csv(file_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(file_2526, parse_dates=['GAME_DATE'])
    combined = pd.concat([df25, df26], ignore_index=True)
    combined['GAME_DATE'] = pd.to_datetime(combined['GAME_DATE'])
    # Ensure DNP column exists
    if 'DNP' not in combined.columns:
        combined['DNP'] = 0
    combined['DNP'] = combined['DNP'].fillna(0).astype(int)
    # CRITICAL: datetime sort, never string sort
    combined = combined.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)
    return combined


def build_player_index(combined_df):
    """
    Build lookup: player_name → full sorted game history (including DNP rows).
    DNP filtering happens at query time via get_prior_games_played().
    """
    pidx = {}
    for pname, grp in combined_df.groupby('PLAYER_NAME', sort=False):
        pidx[pname] = grp.sort_values('GAME_DATE').reset_index(drop=True)
    return pidx


def get_prior_games_played(pidx, player_name, before_date_str):
    """
    Return all games where the player ACTUALLY PLAYED, strictly before the given date.
    DNP rows are excluded — this is the only input to rolling calculations.
    """
    if player_name not in pidx:
        return pd.DataFrame()
    ph = pidx[player_name]
    before = pd.Timestamp(before_date_str)
    prior_all = ph[ph['GAME_DATE'] < before]
    return filter_played(prior_all)


# ─── LIVE FEATURE EXTRACTION ──────────────────────────────────
def extract_prediction_features(prior_played, line):
    """
    Compute all prediction features from prior PLAYED games only.
    prior_played: DataFrame of games where player actually played, sorted chronologically.
    line: the prop line for hit-rate calculations.

    Returns dict of all features, or None if insufficient history (<5 played games).
    """
    if prior_played is None or len(prior_played) < 5:
        return None

    def safe_mean(df, col, n):
        """Mean of last n rows for col, returns None if col missing."""
        if col not in df.columns:
            return None
        vals = df.tail(n)[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else None

    def safe_std(df, col, n, min_n=3):
        if col not in df.columns:
            return 5.0
        vals = df.tail(n)[col].dropna()
        return float(vals.std()) if len(vals) >= min_n else 5.0

    p = prior_played  # alias

    # ── PTS rolling windows ──
    L30 = safe_mean(p, 'PTS', 30) or 0.0
    L20 = safe_mean(p, 'PTS', 20) or L30
    L10 = safe_mean(p, 'PTS', 10) or L30
    L5  = safe_mean(p, 'PTS', 5)  or L30
    L3  = safe_mean(p, 'PTS', 3)  or L30

    # ── FG% (convert to percentage if stored as decimal) ──
    def fg_pct(n):
        v = safe_mean(p, 'FG_PCT', n)
        if v is None: return None
        return round(v * 100, 1) if v < 1.5 else round(v, 1)

    fg30 = fg_pct(30); fg10 = fg_pct(10)
    fgTrend = round(fg10 - fg30, 1) if fg30 is not None and fg10 is not None else None

    # ── FGA ──
    fga30 = safe_mean(p, 'FGA', 30)
    fga10 = safe_mean(p, 'FGA', 10)

    # ── Minutes ──
    m30 = safe_mean(p, 'MIN_NUM', 30)
    m10 = safe_mean(p, 'MIN_NUM', 10)
    minTrend = round(m10 - m30, 1) if m30 is not None and m10 is not None else None

    # ── Variance ──
    std10 = safe_std(p, 'PTS', 10)

    # ── Recent scores (L20 for sparkline) ──
    recent20_pts = list(p.tail(20)['PTS'].fillna(0).astype(int).values)
    recent10_pts = recent20_pts[-10:]
    r20_homes = list(p.tail(20)['IS_HOME'].fillna(0).values.astype(int)) \
                if 'IS_HOME' in p.columns else [0] * len(recent20_pts)

    # ── Hit rates ──
    hr10 = round(sum(1 for r in recent10_pts if r > line) / len(recent10_pts) * 100) \
           if recent10_pts else 50
    hr30 = round(sum(1 for r in recent20_pts if r > line) / len(recent20_pts) * 100) \
           if recent20_pts else 50

    vol   = round(L30 - line, 1)
    trend = round(L5 - L30, 1)

    # ── GBR model features ──
    mp10 = p.tail(10)['MIN_NUM'] if 'MIN_NUM' in p.columns else pd.Series([30.0] * 10)
    mp10 = mp10.fillna(30.0)
    min_cv = float(mp10.std() / mp10.mean()) if mp10.mean() > 0 else 1.0

    pts10 = p.tail(10)['PTS'].fillna(0)
    mn10  = p.tail(10)['MIN_NUM'].replace(0, np.nan).fillna(30.0) if 'MIN_NUM' in p.columns else pd.Series([30.0] * 10)
    ppm   = float((pts10 / mn10).mean()) if len(pts10) > 0 else 0.0

    rmt = float(p.tail(3)['MIN_NUM'].mean() - mp10.mean()) \
          if 'MIN_NUM' in p.columns and len(p) >= 3 else 0.0

    fga10_s = p.tail(10)['FGA'].fillna(0) if 'FGA' in p.columns else pd.Series([0.0] * 10)
    fpm = float((fga10_s / mn10).mean()) if len(fga10_s) > 0 else 0.0

    return {
        'L30': round(L30, 1), 'L20': round(L20, 1),
        'L10': round(L10, 1), 'L5':  round(L5, 1), 'L3': round(L3, 1),
        'fg30': fg30, 'fg10': fg10, 'fgTrend': fgTrend,
        'fga30': round(fga30, 1) if fga30 is not None else None,
        'fga10': round(fga10, 1) if fga10 is not None else None,
        'm30': round(m30, 1) if m30 is not None else None,
        'm10': round(m10, 1) if m10 is not None else None,
        'minTrend': minTrend,
        'std10': round(std10, 1),
        'vol': vol, 'trend': trend,
        'hr10': hr10, 'hr30': hr30,
        'recent20': recent20_pts,
        'recent10': recent10_pts,
        'r20_homes': r20_homes,
        'min_cv':  round(min_cv, 3),
        'ppm':     round(ppm, 3),
        'rmt':     round(rmt, 1),
        'fpm':     round(fpm, 3),
    }


# ─── APPEND ROLLING COLS (for new game log rows) ─────────────
def compute_rolling_for_new_rows(new_df, hist_df):
    """
    Compute L*_* rolling columns for newly appended game rows.
    hist_df: existing history (both seasons combined, already in CSV).
    new_df:  new rows being appended (may include DNP stubs).

    DNP rows: all rolling cols set to NaN (they don't add to the window).
    Played rows: rolling computed from hist played + prior new played rows.

    Returns new_df with all L*_* columns populated.
    Uses explicit loops — never groupby().apply().
    """
    # Ensure DNP column
    if 'DNP' not in new_df.columns: new_df['DNP'] = 0
    new_df['DNP'] = new_df['DNP'].fillna(0).astype(int)
    if 'DNP' not in hist_df.columns: hist_df = hist_df.copy(); hist_df['DNP'] = 0

    # Init all rolling cols as NaN
    for w in WINDOWS:
        for c in ROLL_COLS:
            new_df[f'L{w}_{c}'] = np.nan

    new_df = new_df.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)

    for pname in new_df['PLAYER_NAME'].dropna().unique():
        # Historical played games for this player
        ph_all = hist_df[hist_df['PLAYER_NAME'] == pname].sort_values('GAME_DATE')
        ph_played = filter_played(ph_all)

        # New rows for this player
        pn_all = new_df[new_df['PLAYER_NAME'] == pname].sort_values('GAME_DATE')

        for i, (idx, row) in enumerate(pn_all.iterrows()):
            # DNP rows: leave all rolling cols as NaN
            is_dnp = int(row.get('DNP', 0)) == 1 or float(row.get('MIN_NUM', 0) or 0) == 0
            if is_dnp:
                continue

            # Build prior = historical played + new played rows before this one
            new_played_before = filter_played(pn_all.iloc[:i]) if i > 0 else pd.DataFrame()
            if len(new_played_before) > 0:
                prior = pd.concat([ph_played, new_played_before]).sort_values('GAME_DATE')
            else:
                prior = ph_played.copy()

            for w in WINDOWS:
                subset = prior.tail(w)
                min_req = w // 2 if w >= 100 else min(3, w)
                for c in ROLL_COLS:
                    if c not in prior.columns or len(subset) < min_req:
                        new_df.at[idx, f'L{w}_{c}'] = np.nan
                    else:
                        v = subset[c].mean()
                        new_df.at[idx, f'L{w}_{c}'] = round(float(v), 4) if pd.notna(v) else np.nan

    return new_df
