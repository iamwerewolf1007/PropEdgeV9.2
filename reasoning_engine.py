"""
PropEdge V9.2 — Reasoning Engine
==================================
Generates unique, data-driven pre-match AND post-match reasoning.
Both directions use the same analytical depth and specificity.
No generic templates — every sentence driven by actual signal values.

Post-match uses actual box score data (MIN, FG%, PTS) cross-referenced
against the prediction-time stored values for integrity checking.
"""
import numpy as np
import pandas as pd

# Loss classification labels
LOSS_TYPES = [
    'CLOSE_CALL',        # within 2 pts
    'MINUTES_SHORTFALL', # played significantly fewer minutes
    'SHOOTING_VARIANCE', # normal minutes, hot/cold shooting
    'BLOWOUT_EFFECT',    # game got out of hand, starters pulled
    'MODEL_CORRECT',     # wins only — signals were right
    'MODEL_FAILURE',     # signals were systematically wrong
]

FLAG_NAMES = ['Volume','HR L30','HR L10','Trend','Context','Defense','H2H','Pace','FG Trend','Min Trend']


def _dvp_desc(rank):
    if rank <= 5:  return f"elite defense (#{rank})"
    if rank <= 10: return f"strong defense (#{rank})"
    if rank <= 15: return f"average defense (#{rank})"
    if rank <= 22: return f"below-average defense (#{rank})"
    return f"weak defense (#{rank})"

def _pace_desc(rank):
    if rank <= 5:  return f"fast pace (#{rank})"
    if rank <= 12: return f"above-average pace (#{rank})"
    if rank <= 20: return f"moderate pace (#{rank})"
    return f"slow pace (#{rank})"

def _h2h_avg(h2h_str):
    """Extract numeric average from '22.3 (5g)' format."""
    try: return float(str(h2h_str).split('(')[0].strip())
    except: return None

def _last_name(player):
    parts = str(player).strip().split()
    return parts[-1] if parts else player

def _sign(v):
    return '+' if v >= 0 else ''


# ─── PRE-MATCH REASONING ──────────────────────────────────────
def generate_pre_match_reason(play):
    """
    Generate unique pre-match reasoning from play data.
    5-part structure: lead → matchup context → signal audit → model → risk.
    """
    direction = play.get('dir', 'OVER')
    is_over   = 'UNDER' not in direction
    is_lean   = 'LEAN'  in direction
    line      = float(play.get('line', 0) or 0)
    player    = play.get('player', '')
    name      = _last_name(player)

    L30   = float(play.get('l30', 0) or 0)
    L10   = float(play.get('l10', L30) or L30)
    L5    = float(play.get('l5',  L30) or L30)
    L3    = float(play.get('l3',  L30) or L30)
    vol   = float(play.get('volume', L30 - line) or 0)
    trend = float(play.get('trend',  L5 - L30)   or 0)
    std10 = float(play.get('std10', 5.0) or 5.0)
    flags     = int(play.get('flags', 0) or 0)
    flag_details = play.get('flagDetails', [])
    conf      = float(play.get('conf', 0.55) or 0.55)
    defP      = int(play.get('defP', 15) or 15)
    pace      = int(play.get('pace', 15) or 15)
    fgTrend   = play.get('fgTrend')
    minTrend  = play.get('minTrend')
    m10       = play.get('minL10')
    m30       = play.get('minL30')
    hr30      = int(play.get('hr30', 50) or 50)
    hr10      = int(play.get('hr10', 50) or 50)
    recent    = play.get('recent', [])
    predPts   = play.get('predPts')
    predGap   = play.get('predGap')
    h2h_str   = play.get('h2h', '')
    h2hG      = int(play.get('h2hG', 0) or 0)
    h2hTsDev  = float(play.get('h2hTsDev', 0) or 0)
    tl        = play.get('tierLabel', 'T3')

    h2h_avg_pts = _h2h_avg(h2h_str) if h2hG >= 3 else None
    use_h2h = h2h_avg_pts is not None and h2hG >= 3

    agrees    = [f for f in flag_details if f.get('agrees')]
    disagrees = [f for f in flag_details if not f.get('agrees')]
    agree_names    = [f['name'] for f in agrees]
    disagree_names = [f['name'] for f in disagrees]

    parts = []

    # ── S1: Lead with strongest evidence ──
    # Score each candidate signal for lead priority
    candidates = []
    if abs(vol) >= 2:
        sup = (is_over and vol > 0) or (not is_over and vol < 0)
        candidates.append((abs(vol) * (1.5 if sup else 1.1), 'vol',
            f"{name}'s L30 of {L30:.1f} pts sits {abs(vol):.1f} {'above' if vol > 0 else 'below'} the {line} line"))
    if use_h2h and h2h_avg_pts is not None and abs(h2h_avg_pts - line) >= 2:
        sup = (is_over and h2h_avg_pts > line) or (not is_over and h2h_avg_pts < line)
        gap = h2h_avg_pts - line
        candidates.append((abs(gap) * (1.4 if sup else 1.1), 'h2h',
            f"{name} averages {h2h_avg_pts:.1f} pts in {h2hG} meetings ({_sign(gap)}{gap:.1f} vs line)"))
    if abs(trend) >= 2:
        sup = (is_over and trend > 0) or (not is_over and trend < 0)
        candidates.append((abs(trend) * (1.3 if sup else 1.0), 'trend',
            f"recent scoring {'trending up' if trend > 0 else 'trending down'} {abs(trend):.1f} pts (L5 vs L30)"))
    if std10 <= 5:
        candidates.append((6 - std10, 'consistency',
            f"low scoring variance (σ={std10:.1f}) — highly predictable output near line"))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        lead = candidates[0][2]
    else:
        lead = f"{name}'s L30 average of {L30:.1f} pts vs the {line} line"

    dir_word = direction.replace('LEAN ', 'lean ')
    parts.append(f"{lead[0].upper()}{lead[1:]} — {'supports' if flags >= 6 else 'marginally supports'} the {dir_word}.")

    # ── S2: Matchup and efficiency context ──
    ctx = []
    if use_h2h and h2h_avg_pts is not None and abs(h2h_avg_pts - L30) >= 1.5:
        diff = h2h_avg_pts - L30
        ctx.append(f"H2H avg {abs(diff):.1f} pts {'above' if diff > 0 else 'below'} season L30")
    if abs(h2hTsDev) >= 3 and use_h2h:
        ctx.append(f"TS% shifts {_sign(h2hTsDev)}{h2hTsDev:.1f}% in this matchup")
    if fgTrend is not None and abs(fgTrend) >= 3:
        ctx.append(f"FG% {_sign(fgTrend)}{fgTrend:.1f}% trending (L10 vs L30)")
    if m10 is not None and m30 is not None and abs(m10 - m30) >= 2:
        ctx.append(f"minutes {'up' if m10 > m30 else 'down'} {abs(m10-m30):.1f} recently (L10={m10:.1f} vs L30={m30:.1f})")

    def_str  = _dvp_desc(defP)
    pace_str = _pace_desc(pace)
    matchup  = f"Opponent is {def_str} at {pace_str}."
    if ctx:
        parts.append(f"{'; '.join(c[0].upper()+c[1:] for c in ctx[:2])}. {matchup}")
    else:
        parts.append(matchup)

    # ── S3: Named signal audit ──
    if flags >= 8 and len(disagrees) == 0:
        parts.append(f"Full consensus: {flags}/10 signals aligned ({', '.join(agree_names[:4])}).")
    elif flags >= 6:
        agree_str    = ', '.join(agree_names[:4]) if agree_names else 'none'
        disagree_str = ', '.join(disagree_names[:3]) if disagree_names else 'none'
        parts.append(f"{flags}/10 signals agree — {agree_str} support the {dir_word.split()[-1].upper()}; {disagree_str} {'dissent' if len(disagrees) > 1 else 'dissents'}.")
    else:
        agree_str    = ', '.join(agree_names[:3]) if agree_names else 'none'
        disagree_str = ', '.join(disagree_names[:3]) if disagree_names else 'none'
        parts.append(f"Mixed signals: {flags}/10 agree ({agree_str}); counter-signals: {disagree_str}.")

    # ── S4: Model projection ──
    if predPts is not None:
        gap_dir = 'above' if predPts > line else 'below'
        parts.append(
            f"Projection model targets {predPts:.1f} pts ({predGap:.1f} pts {gap_dir} line; "
            f"{int(conf*100)}% blended confidence [{tl}])."
        )

    # ── S5: Specific risk ──
    risks = []
    l3_vs_l30 = L3 - L30
    if is_over and l3_vs_l30 < -4:
        risks.append(f"L3 has dropped to {L3:.1f} ({abs(l3_vs_l30):.1f} below L30) — "
                     f"deepening slump makes the over vulnerable")
    elif not is_over and l3_vs_l30 > 4:
        last = recent[0] if recent else None
        extra = f" after a {last}-pt outing" if last else ""
        risks.append(f"L3 has surged to {L3:.1f}{extra} ({l3_vs_l30:.1f} above L30) — "
                     f"momentum could push over the under line")
    if std10 > 7 and not risks:
        risks.append(f"high variance (σ={std10:.1f}) makes the outcome difficult to call with confidence")
    if is_over and hr30 < 42 and not risks:
        risks.append(f"only {hr30}% hit rate over L30 suggests the line may be set fairly")
    elif not is_over and hr30 > 58 and not risks:
        risks.append(f"{hr30}% over-rate on L30 suggests scoring tendency that could threaten the under")
    if m10 is not None and m30 is not None and m10 - m30 < -3 and not risks:
        risks.append(f"minutes trending down ({m10:.1f} L10 vs {m30:.1f} L30) — "
                     f"role reduction would suppress counting stats")

    if risks:
        parts.append(f"Risk: {risks[0]}.")

    result = ' '.join(p for p in parts if p.strip())
    if is_lean:
        result = '[Low conviction — lean only] ' + result
    return result


# ─── POST-MATCH REASONING ─────────────────────────────────────
def generate_post_match_reason(play, actual_box=None):
    """
    Generate post-match reasoning after result is known.

    play:       the play dict (contains stored prediction-time fields)
    actual_box: dict with actual game stats: {
                    'actual_pts': int,
                    'actual_min': float,
                    'actual_fgm': int,
                    'actual_fga': int,
                    'actual_fg_pct': float,  # as percentage e.g. 45.2
                    'integrity_flag': str or None  # if rolling stats deviated
                }
                If None, falls back to play dict fields only.

    Returns (reason_str, loss_type_str)
    """
    direction  = play.get('dir', 'OVER')
    is_over    = 'UNDER' not in direction
    result     = play.get('result', '')
    line       = float(play.get('line', 0) or 0)
    player     = play.get('player', '')
    name       = _last_name(player)
    flag_details = play.get('flagDetails', [])

    # Actual values — prefer actual_box, fall back to play dict
    if actual_box:
        actual_pts  = actual_box.get('actual_pts')
        actual_min  = actual_box.get('actual_min')
        actual_fgm  = actual_box.get('actual_fgm')
        actual_fga  = actual_box.get('actual_fga')
        actual_fg_pct = actual_box.get('actual_fg_pct')
        integrity_flag = actual_box.get('integrity_flag')
    else:
        actual_pts  = play.get('actualPts')
        actual_min  = None
        actual_fgm  = None
        actual_fga  = None
        actual_fg_pct = None
        integrity_flag = None

    # Prediction-time stored values
    stored_m10   = play.get('minL10')
    stored_fg10  = play.get('fgL10')
    stored_l30   = float(play.get('l30', 0) or 0)
    pred_pts     = play.get('predPts')
    flags        = int(play.get('flags', 0) or 0)
    delta        = play.get('delta', 0) or 0

    if actual_pts is None or result not in ('WIN', 'LOSS'):
        return '', None

    actual_pts = int(actual_pts)
    delta      = round(actual_pts - line, 1)
    margin     = abs(delta)

    agrees    = [f for f in flag_details if f.get('agrees')]
    disagrees = [f for f in flag_details if not f.get('agrees')]
    agree_names    = [f['name'] for f in agrees]
    disagree_names = [f['name'] for f in disagrees]

    parts = []
    loss_type = None

    # ── S1: Outcome statement ──
    dir_word = direction.replace('LEAN ', '')
    outcome  = 'hit' if result == 'WIN' else 'missed'
    parts.append(
        f"{'WIN' if result=='WIN' else 'LOSS'} — {name} scored {actual_pts} pts vs {line} line "
        f"({_sign(delta)}{delta:.1f}), {dir_word.upper()} {outcome} by {margin:.1f} pts."
    )

    # ── S2: Minutes analysis ──
    if actual_min is not None and stored_m10 is not None:
        min_diff  = actual_min - stored_m10
        min_pct   = abs(min_diff) / stored_m10 * 100 if stored_m10 > 0 else 0
        if abs(min_diff) >= 3:
            direction_word = 'above' if min_diff > 0 else 'below'
            parts.append(
                f"{name} played {actual_min:.0f} min ({_sign(min_diff)}{min_diff:.1f} vs L10 avg of "
                f"{stored_m10:.1f}) — {min_pct:.0f}% {'more' if min_diff > 0 else 'less'} than expected."
            )
            if min_diff < -4 and result == 'LOSS':
                loss_type = 'MINUTES_SHORTFALL'
        else:
            parts.append(
                f"Minutes were as expected: {actual_min:.0f} played vs L10 avg {stored_m10:.1f}."
            )
    elif actual_min is not None:
        parts.append(f"Played {actual_min:.0f} minutes.")

    # ── S3: Efficiency analysis ──
    if actual_fga is not None and actual_fgm is not None:
        if actual_fga > 0:
            fg_actual = round(actual_fgm / actual_fga * 100, 1)
        else:
            fg_actual = 0.0

        if stored_fg10 is not None:
            fg_diff = fg_actual - stored_fg10
            if abs(fg_diff) >= 5:
                eff_word = 'hot' if fg_diff > 0 else 'cold'
                parts.append(
                    f"Shooting efficiency was {eff_word}: {fg_actual:.1f}% FG "
                    f"({_sign(fg_diff)}{fg_diff:.1f}% vs L10 avg of {stored_fg10:.1f}%). "
                    f"{actual_fgm}/{actual_fga} from the field."
                )
                if result == 'LOSS' and loss_type is None:
                    loss_type = 'SHOOTING_VARIANCE'
            else:
                parts.append(
                    f"Shooting was in line with averages: {fg_actual:.1f}% FG "
                    f"({actual_fgm}/{actual_fga}, L10 avg {stored_fg10:.1f}%)."
                )
        else:
            if actual_fga > 0:
                parts.append(f"Shot {actual_fgm}/{actual_fga} ({fg_actual:.1f}% FG).")

    # ── S4: Signal audit — which signals were right/wrong ──
    if agree_names or disagree_names:
        if result == 'WIN':
            correct_signals = agree_names
            wrong_signals   = disagree_names
            if correct_signals:
                parts.append(
                    f"Signals that called it correctly: {', '.join(correct_signals[:5])} "
                    f"({len(correct_signals)}/10 aligned)."
                )
            if wrong_signals:
                parts.append(
                    f"Counter-signals that were wrong: {', '.join(wrong_signals[:3])}."
                )
        else:
            # For a LOSS, the disagreeing signals proved correct
            if disagree_names:
                parts.append(
                    f"Counter-signals that proved correct: {', '.join(disagree_names[:4])}. "
                    f"Signals that were wrong: {', '.join(agree_names[:3]) if agree_names else 'none'}."
                )
            else:
                parts.append(
                    f"All {flags} agreeing signals ({', '.join(agree_names[:4])}) were wrong — "
                    f"result went directly against the consensus."
                )

    # ── S5: Model accuracy ──
    if pred_pts is not None:
        model_err = abs(actual_pts - pred_pts)
        model_dir_correct = (is_over and pred_pts > line) or (not is_over and pred_pts < line)
        parts.append(
            f"Projection model targeted {pred_pts:.1f} pts — "
            f"actual {actual_pts} was {model_err:.1f} pts {'off' if model_err > 2 else 'close'} "
            f"({'correct direction' if model_dir_correct else 'wrong direction'})."
        )

    # ── S6: Loss classification (if not already set) ──
    if loss_type is None:
        if margin <= 2:
            loss_type = 'CLOSE_CALL'
        elif result == 'WIN':
            loss_type = 'MODEL_CORRECT'
        else:
            # Blowout: large margin, wrong way
            if (is_over and actual_pts < line - 8) or (not is_over and actual_pts > line + 8):
                loss_type = 'BLOWOUT_EFFECT'
            elif flags >= 7 and margin > 3:
                # Strong signal consensus but still lost — variance
                loss_type = 'SHOOTING_VARIANCE'
            else:
                loss_type = 'MODEL_FAILURE'

    if result == 'LOSS':
        parts.append(f"Loss classification: {loss_type}.")
    else:
        parts.append(f"Result classification: {loss_type}.")

    # ── S7: Learning note ──
    opp = play.get('match', '').split(' @ ')
    opp_team = opp[-1] if play.get('isHome') else (opp[0] if len(opp) > 1 else '')
    if result == 'WIN':
        if flags >= 8:
            parts.append(
                f"Note: Strong signal alignment ({flags}/10) with confirmed result — "
                f"this player/matchup pattern is reliable for future predictions."
            )
        elif use_model_correct := (pred_pts is not None and abs(actual_pts - pred_pts) <= 2):
            parts.append(
                f"Note: Projection model accuracy within 2 pts — GBR features "
                f"are well-calibrated for {name} in this role."
            )
        else:
            parts.append(
                f"Note: Result confirms {name}'s consistency near this line level."
            )
    else:
        if loss_type == 'CLOSE_CALL':
            parts.append(
                f"Note: Within 2 pts — direction was sound. "
                f"Small execution variance, no model adjustment needed."
            )
        elif loss_type == 'MINUTES_SHORTFALL':
            parts.append(
                f"Note: Minutes were the deciding factor, not scoring rate. "
                f"Flag {name} for minutes risk in future props if role uncertainty exists."
            )
        elif loss_type == 'SHOOTING_VARIANCE':
            parts.append(
                f"Note: Volume and opportunity signals were correct but shooting "
                f"efficiency diverged. Normal variance — per-minute output was on track."
            )
        elif loss_type == 'BLOWOUT_EFFECT':
            parts.append(
                f"Note: Game script invalidated the prop. "
                f"Consider blowout probability when {opp_team} is involved."
            )
        else:
            parts.append(
                f"Note: Model signals were misleading for this matchup — "
                f"review {name}'s recent role and usage for pattern shifts."
            )

    # ── Integrity flag ──
    if integrity_flag:
        parts.append(f"⚠ Data integrity: {integrity_flag}")

    reason = ' '.join(p for p in parts if p.strip())
    return reason, loss_type
