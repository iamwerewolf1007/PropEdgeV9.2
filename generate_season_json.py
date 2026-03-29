#!/usr/bin/env python3
"""
PropEdge V9.2 — Generate Season JSONs
========================================
2024-25: Synthetic lines from game logs (all players with ≥10 games)
2025-26: ALL rows from real prop lines Excel — zero filtering.
         DNP tagged, graded where actual PTS available in game log.
         Missing box scores (e.g. Mar 28): plays remain ungraded.
"""
import pandas as pd
import numpy as np
import json, sys, time, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import *
from audit import log_event
from rolling_engine import load_combined, build_player_index, get_prior_games_played, extract_prediction_features
from reasoning_engine import generate_pre_match_reason, generate_post_match_reason


def _s(v):
    """Safe scalar for JSON."""
    if v is None: return None
    if isinstance(v, float) and np.isnan(v): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 4)
    if isinstance(v, np.bool_): return bool(v)
    if isinstance(v, pd.Timestamp): return v.strftime('%Y-%m-%d')
    return v


def run_model_on_props(props_df, all_logs, h2h_df, model, player_trust, season_label):
    """
    Run full prediction engine on a props DataFrame.
    ALL rows are processed — no filtering by tier/confidence/direction.
    DNP: play record created with result='DNP' if player had no box score.
    Ungraded: result=None if no actual PTS available.
    """
    from rolling_engine import filter_played

    # Combined game logs sorted
    logs = all_logs.copy()
    logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
    if 'DNP' not in logs.columns: logs['DNP'] = 0
    logs = logs.sort_values(['PLAYER_NAME','GAME_DATE']).reset_index(drop=True)

    # Deduplicate H2H rows — keep last entry per (PLAYER_NAME, OPPONENT) pair
    # This prevents a Series being returned when there are duplicate keys,
    # which causes "truth value of a Series is ambiguous" on `if hr_`
    h2h_dedup = h2h_df.drop_duplicates(subset=['PLAYER_NAME','OPPONENT'], keep='last')
    h2h_lkp   = {(r['PLAYER_NAME'], r['OPPONENT']): r.to_dict()
                 for _, r in h2h_dedup.iterrows()}
    pidx     = build_player_index(logs)

    played_only = filter_played(logs)
    team_fga  = played_only.groupby('OPPONENT')['FGA'].mean()
    pace_rank = {t: i+1 for i,(t,_) in enumerate(team_fga.sort_values(ascending=False).items())}

    # Actual results from game logs (played rows only, with PTS > 0)
    results_lkp = {}  # (player, date_str) → row dict
    box_lkp     = {}  # (player, date_str) → {pts, min, fgm, fga, fg_pct}
    for _, r in played_only.iterrows():
        k = (r['PLAYER_NAME'], r['GAME_DATE'].strftime('%Y-%m-%d'))
        results_lkp[k] = int(r['PTS']) if pd.notna(r.get('PTS')) else None
        if pd.notna(r.get('PTS')):
            fgm = int(r.get('FGM',0) or 0)
            fga = int(r.get('FGA',0) or 0)
            raw_fg = r.get('FG_PCT',0) or 0
            fg_pct = round(float(raw_fg)*100,1) if float(raw_fg)<1.5 else round(float(raw_fg),1)
            box_lkp[k] = {
                'actual_pts': int(r['PTS']),
                'actual_min': round(float(r.get('MIN_NUM',0) or 0),1),
                'actual_fgm': fgm,
                'actual_fga': fga,
                'actual_fg_pct': fg_pct,
                'integrity_flag': None,
            }

    # B2B map
    b2b_map = {}
    for pn, g in played_only.sort_values('GAME_DATE').groupby('PLAYER_NAME'):
        dates = g['GAME_DATE'].values
        for i in range(len(dates)):
            ds = pd.Timestamp(dates[i]).strftime('%Y-%m-%d')
            b2b_map[(pn,ds)] = int((dates[i]-dates[i-1]).astype('timedelta64[D]').astype(int)) if i>0 else 99

    plays = []; skipped = 0; processed = 0
    total = len(props_df)

    for _, prop in props_df.iterrows():
        processed += 1
        if processed % 2000 == 0:
            print(f"    {processed}/{total}...")

        player    = str(prop.get('Player','')).strip()
        date_str  = prop['Date'].strftime('%Y-%m-%d') if hasattr(prop['Date'],'strftime') else str(prop['Date'])[:10]
        line      = prop.get('Line')
        game      = str(prop.get('Game',''))
        home_team = str(prop.get('Home',''))
        away_team = str(prop.get('Away',''))
        raw_pos   = str(prop.get('Position','') or '')
        position  = POS_MAP.get(raw_pos, 'Forward')

        if not player or pd.isna(line):
            skipped += 1; continue
        line = float(line)

        # Live rolling from played games only
        prior = get_prior_games_played(pidx, player, date_str)

        # If no history at all → still include play but mark with minimal data
        if len(prior) < 5:
            # Check if there's any result to grade
            k = (player, date_str)
            actual_pts = results_lkp.get(k)
            plays.append(_make_minimal_play(
                player, date_str, line, game, home_team, away_team,
                position, actual_pts, season_label,
                prop.get('Over Odds'), prop.get('Under Odds'), prop.get('Books',1)
            ))
            continue

        feats = extract_prediction_features(prior, line)
        if feats is None:
            k = (player, date_str)
            plays.append(_make_minimal_play(
                player, date_str, line, game, home_team, away_team,
                position, results_lkp.get(k), season_label,
                prop.get('Over Odds'), prop.get('Under Odds'), prop.get('Books',1)
            ))
            continue

        L30=feats['L30']; L20=feats['L20']; L10=feats['L10']
        L5=feats['L5'];   L3=feats['L3']
        vol=feats['vol']; trend=feats['trend']; std10=feats['std10']
        hr10=feats['hr10']; hr30=feats['hr30']
        r20=feats['recent20']; r20h=feats['r20_homes']
        fg30=feats['fg30']; fg10=feats['fg10']; fgTrend=feats['fgTrend']
        m30=feats['m30'];   m10=feats['m10'];   minTrend=feats['minTrend']
        fga30=feats['fga30']; fga10=feats['fga10']
        min_cv=feats['min_cv']; ppm=feats['ppm']; rmt=feats['rmt']; fpm=feats['fpm']

        sn       = prior.iloc[-1]
        team_abr = str(sn.get('GAME_TEAM_ABBREVIATION',''))
        is_home  = team_abr == home_team
        opp      = away_team if is_home else home_team
        if not position or position == 'Forward':
            snap_pos = POS_MAP.get(str(sn.get('PLAYER_POSITION','')), 'Forward')
            if snap_pos: position = snap_pos

        hr_     = h2h_lkp.get((player, opp))
        h2hG    = int(hr_['H2H_GAMES'])           if hr_ else 0
        h2h_avg = float(hr_['H2H_AVG_PTS'])       if hr_ else None
        h2h_ts  = float(hr_['H2H_TS_VS_OVERALL'])   if hr_ and pd.notna(hr_.get('H2H_TS_VS_OVERALL'))   else 0
        h2h_fga = float(hr_['H2H_FGA_VS_OVERALL'])  if hr_ and pd.notna(hr_.get('H2H_FGA_VS_OVERALL'))  else 0
        h2h_min = float(hr_['H2H_MIN_VS_OVERALL'])  if hr_ and pd.notna(hr_.get('H2H_MIN_VS_OVERALL'))  else 0
        h2h_cf  = float(hr_['H2H_CONFIDENCE'])      if hr_ and pd.notna(hr_.get('H2H_CONFIDENCE'))      else 0
        h2h_str = f"{h2h_avg:.1f} ({h2hG}g)"  if h2hG >= 3 and h2h_avg else ""
        use_h2h = h2hG >= 3 and h2h_avg is not None

        defP = get_dvp(opp, position); defO = get_def_overall(opp)
        op   = pace_rank.get(opp, 15)
        rest = b2b_map.get((player, date_str), 99); ib2b = 1 if rest==1 else 0

        # GBR projection
        pred_pts = None; pred_gap = 0
        if model is not None:
            from model_trainer import FEATURES
            fd = {'l30':L30,'l10':L10,'l5':L5,'l3':L3,'volume':vol,'trend':trend,
                  'std10':std10,'defP':defP,'pace_rank':op,'h2h_ts_dev':h2h_ts,
                  'h2h_fga_dev':h2h_fga,'h2h_min_dev':h2h_min,'h2h_conf':h2h_cf,
                  'min_cv':min_cv,'pts_per_min':ppm,'recent_min_trend':rmt,
                  'fga_per_min':fpm,'is_b2b':ib2b,'rest_days':rest,
                  'consistency':1/(std10+1),'line':line}
            Xp = pd.DataFrame([fd])[FEATURES].fillna(0)
            pred_pts = float(model.predict(Xp)[0]); pred_gap = abs(pred_pts - line)

        # 10-signal composite
        W = POS_WEIGHTS.get(position, POS_WEIGHTS['Forward'])
        S = {
            1:np.clip((L30-line)/5,-1,1), 2:(hr30/100-0.5)*2, 3:(hr10/100-0.5)*2,
            4:np.clip((L5-L30)/5,-1,1),   5:np.clip(vol/5,-1,1),
            6:np.clip((defP-15)/15,-1,1),
            7:np.clip((h2h_avg-line)/5,-1,1) if use_h2h else 0.0,
            8:np.clip((15-op)/15,-1,1),
            9:np.clip((fgTrend or 0)/10,-1,1), 10:np.clip((minTrend or 0)/5,-1,1),
        }
        if use_h2h:
            tw=sum(W.values()); ws=sum(W[k]*S[k] for k in S)
        else:
            tw=sum(v for k,v in W.items() if k!=7); ws=sum(W[k]*S[k] for k in S if k!=7)
        composite = ws/tw if tw else 0

        if pred_pts is not None:
            if pred_pts > line+0.3:   direction='OVER';  is_lean=False
            elif pred_pts < line-0.3: direction='UNDER'; is_lean=False
            else:
                direction=f"LEAN {'OVER' if pred_pts>=line else 'UNDER'}"; is_lean=True
        else:
            if composite > 0.05:   direction='OVER';  is_lean=False
            elif composite < -0.05: direction='UNDER'; is_lean=False
            else:
                direction=f"LEAN {'OVER' if composite>=0 else 'UNDER'}"; is_lean=True

        sc   = float(np.clip(0.5+abs(composite)*0.3,0.50,0.85))
        if std10>8: sc-=0.03
        sc   = float(np.clip(sc,0.45,0.85))
        pc   = float(np.clip(0.5+pred_gap*0.04,0.45,0.90)) if pred_pts else sc
        conf = 0.4*sc + 0.6*pc

        io = 'UNDER' not in direction
        flags=0; fds=[]
        for nm,ag,dt in [
            ('Volume',  (io and vol>0) or (not io and vol<0),                   f"{vol:+.1f}"),
            ('HR L30',  (io and hr30>50) or (not io and hr30<50),               f"{hr30}%"),
            ('HR L10',  (io and hr10>50) or (not io and hr10<50),               f"{hr10}%"),
            ('Trend',   (io and trend>0) or (not io and trend<0),               f"{trend:+.1f}"),
            ('Context', (io and vol>-1)  or (not io and vol<1),                 f"vol={vol:+.1f}"),
            ('Defense', (io and defP>15) or (not io and defP<15),              f"#{defP}"),
            ('H2H',     use_h2h and ((io and h2h_avg>line) or (not io and h2h_avg<line)), f"{h2h_avg:.1f}" if use_h2h else "N/A"),
            ('Pace',    (io and op<15)   or (not io and op>15),                 f"#{op}"),
            ('FG Trend',fgTrend is not None and ((io and fgTrend>0) or (not io and fgTrend<0)), f"{fgTrend:+.1f}%" if fgTrend else "N/A"),
            ('Min Trend',minTrend is not None and ((io and minTrend>0) or (not io and minTrend<0)), f"{minTrend:+.1f}" if minTrend else "N/A"),
        ]:
            flags += 1 if ag else 0
            fds.append({'name':nm,'agrees':bool(ag),'detail':dt})

        h2h_aligned = True
        if h2h_ts != 0:
            if 'OVER'  in direction and h2h_ts < -3: h2h_aligned=False
            elif 'UNDER' in direction and h2h_ts >  3: h2h_aligned=False

        if is_lean:             tier=3; tl='T3_LEAN'
        elif conf>=0.70 and flags>=8 and std10<=6 and h2h_aligned: tier=1; tl='T1_ULTRA'
        elif conf>=0.65 and flags>=7 and std10<=7 and h2h_aligned: tier=1; tl='T1_PREMIUM'
        elif conf>=0.62 and flags>=7 and std10<=7 and h2h_aligned: tier=1; tl='T1'
        elif conf>=0.55 and flags>=6 and std10<=8 and h2h_aligned: tier=2; tl='T2'
        else:                   tier=3; tl='T3'

        tr = player_trust.get(player)
        if tr is not None and tr < 0.42 and tier==1: tier=2; tl='T2'
        units = 3.0 if tl=='T1_ULTRA' else 2.0 if tier==1 else 1.0 if tier==2 else 0.0

        over_odds  = american_to_decimal(prop.get('Over Odds'))
        under_odds = american_to_decimal(prop.get('Under Odds'))
        ro = sum(1 for r in r20 if r > line)
        ru = sum(1 for r in r20 if r <= line)

        # Pre-match reasoning
        pd_data = {
            'player':player,'dir':direction,'line':line,
            'l30':L30,'l10':L10,'l5':L5,'l3':L3,
            'volume':vol,'trend':trend,'std10':std10,
            'flags':flags,'flagDetails':fds,
            'h2h':h2h_str,'h2hG':h2hG,'h2hTsDev':h2h_ts,'h2hFgaDev':h2h_fga,
            'h2hProfile':hr_.get('H2H_SCORING_PROFILE','') if hr_ else '',
            'defP':defP,'defO':defO,'pace':op,
            'fgTrend':fgTrend,'minTrend':minTrend,'minL30':m30,'minL10':m10,
            'conf':conf,'predPts':round(pred_pts,1) if pred_pts else None,
            'predGap':round(pred_gap,1) if pred_pts else None,
            'tierLabel':tl,'position':position,'match':game,'isHome':is_home,
            'recent':r20[:5],'hr30':hr30,'hr10':hr10,
        }
        pre_reason = generate_pre_match_reason(pd_data)

        # Grade + post-match reasoning
        k = (player, date_str)
        actual_pts = results_lkp.get(k)
        box_data   = box_lkp.get(k)

        if actual_pts is not None:
            actual_pts = int(actual_pts)
            if 'OVER'  in direction: result='WIN' if actual_pts>line else 'LOSS'
            elif 'UNDER' in direction: result='WIN' if actual_pts<line else 'LOSS'
            else: result='NO PLAY'
            delta = round(actual_pts - line, 1)
        else:
            result    = None
            delta     = None

        post_reason = ''; loss_type = None
        if result in ('WIN','LOSS') and box_data:
            play_for_post = {**pd_data, 'actualPts':actual_pts, 'result':result, 'delta':delta}
            post_reason, loss_type = generate_post_match_reason(play_for_post, box_data)

        plays.append({
            'date':date_str,'player':player,'match':game,'fullMatch':game,
            'isHome':is_home,'team':team_abr,'gameTime':'','position':position,'posSimple':position[:1],
            'line':_s(line),'overOdds':_s(over_odds),'underOdds':_s(under_odds),
            'books':_s(prop.get('Books',1)),'spread':None,'total':None,'blowout':False,
            'l30':_s(round(L30,1)),'l20':_s(round(L20,1)),'l10':_s(round(L10,1)),
            'l5':_s(round(L5,1)),'l3':_s(round(L3,1)),
            'hr30':hr30,'hr10':hr10,
            'recent':r20[:5],'recent10':r20[:10],'recent20':r20,
            'recent20homes':[bool(x) for x in r20h],
            'defO':defO,'defP':defP,'pace':op,
            'h2h':h2h_str,'h2hG':h2hG,'h2hTsDev':_s(h2h_ts),'h2hFgaDev':_s(h2h_fga),
            'h2hConfidence':_s(h2h_cf),'h2hProfile':hr_.get('H2H_SCORING_PROFILE','') if hr_ else '',
            'fgL30':_s(fg30),'fgL10':_s(fg10),'fga30':_s(fga30),'fga10':_s(fga10),
            'fg3L30':None,'fg3L10':None,'minL30':_s(m30),'minL10':_s(m10),'std10':round(std10,1),
            'dir':direction,'rawDir':direction,'conf':round(conf,3),
            'tier':tier,'tierLabel':tl,'units':units,'avail':'OK',
            'volume':vol,'trend':trend,'fgTrend':_s(fgTrend),'minTrend':_s(minTrend),
            'flags':flags,'flagsStr':f"{flags}/10",'flagDetails':fds,
            'recentOver':ro,'recentUnder':ru,
            'lineHistory':[{'line':_s(line),'batch':0,'ts':''}],
            'predPts':_s(round(pred_pts,1)) if pred_pts else None,
            'predGap':_s(round(pred_gap,1)) if pred_pts else None,
            'preMatchReason':pre_reason,
            'actualPts':_s(actual_pts),'result':result,'delta':_s(delta),
            'postMatchReason':post_reason,'lossType':loss_type,'reason':'',
            'season':season_label,
        })

    print(f"    Processed {processed}, built {len(plays)} play records, {skipped} skipped (missing player/line data)")
    return plays


def _make_minimal_play(player, date_str, line, game, home_team, away_team,
                        position, actual_pts, season_label, over_odds, under_odds, books):
    """Create a minimal play record for players with insufficient history."""
    result = delta = None
    if actual_pts is not None:
        # Can't grade without direction — mark as NO PLAY
        result = 'NO PLAY'; delta = None
    return {
        'date':date_str,'player':player,'match':game,'fullMatch':game,
        'isHome':game.split(' @ ')[-1] == home_team if ' @ ' in game else False,
        'team':'','gameTime':'','position':position,'posSimple':position[:1],
        'line':_s(line),'overOdds':_s(american_to_decimal(over_odds)),
        'underOdds':_s(american_to_decimal(under_odds)),
        'books':_s(books),'spread':None,'total':None,'blowout':False,
        'l30':None,'l20':None,'l10':None,'l5':None,'l3':None,
        'hr30':None,'hr10':None,'recent':[],'recent10':[],'recent20':[],
        'recent20homes':[],'defO':None,'defP':None,'pace':None,
        'h2h':'','h2hG':0,'h2hTsDev':0,'h2hFgaDev':0,'h2hConfidence':0,'h2hProfile':'',
        'fgL30':None,'fgL10':None,'fga30':None,'fga10':None,'fg3L30':None,'fg3L10':None,
        'minL30':None,'minL10':None,'std10':None,
        'dir':None,'rawDir':None,'conf':None,'tier':3,'tierLabel':'T3','units':0,'avail':'INSUFFICIENT_HISTORY',
        'volume':None,'trend':None,'fgTrend':None,'minTrend':None,
        'flags':0,'flagsStr':'0/10','flagDetails':[],
        'recentOver':0,'recentUnder':0,'lineHistory':[{'line':_s(line),'batch':0,'ts':''}],
        'predPts':None,'predGap':None,
        'preMatchReason':'Insufficient game history for analysis.',
        'actualPts':_s(actual_pts),'result':result,'delta':_s(delta),
        'postMatchReason':'','lossType':None,'reason':'','season':season_label,
    }


def main():
    print("="*60)
    print("PropEdge V9.2 — Generate Season JSONs")
    print("="*60)
    t0 = time.time()

    df25 = pd.read_csv(FILE_GL_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(FILE_GL_2526, parse_dates=['GAME_DATE'])
    h2h  = pd.read_csv(FILE_H2H)
    all_logs = pd.concat([df25, df26], ignore_index=True)
    all_logs['GAME_DATE'] = pd.to_datetime(all_logs['GAME_DATE'])

    # Load or train model
    model = None
    if FILE_MODEL.exists():
        with open(FILE_MODEL,'rb') as f: model = pickle.load(f)
        print("  ✓ Loaded existing model")
    else:
        print("  Training model first...")
        from model_trainer import train_and_save
        model = train_and_save(FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_MODEL, FILE_TRUST)

    trust = {}
    if FILE_TRUST.exists():
        with open(FILE_TRUST) as f: trust = json.load(f)

    # ── 2024-25: Synthetic lines ──────────────────────────────
    print("\n  Generating 2024-25 synthetic props (all players ≥10 games)...")
    from synthetic_lines import generate_season_lines
    synth = generate_season_lines(df25, '2024-25')
    print(f"  Synthetic props: {len(synth)}")

    print("  Running model on 2024-25...")
    plays_25 = run_model_on_props(synth, all_logs, h2h, model, trust, '2024-25')
    graded_25 = [p for p in plays_25 if p['result'] in ('WIN','LOSS')]
    wins_25   = sum(1 for p in graded_25 if p['result']=='WIN')
    pct_25    = f"{wins_25/len(graded_25)*100:.1f}%" if graded_25 else "—"
    print(f"  2024-25: {len(plays_25)} plays, {len(graded_25)} graded, {wins_25}W/{len(graded_25)-wins_25}L = {pct_25}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEASON_2425,'w') as f: json.dump(clean_json(plays_25), f)
    print(f"  ✓ Saved {SEASON_2425.name} ({len(plays_25)} plays) — LOCKED")
    log_event('GEN','SEASON_2425_GENERATED',SEASON_2425.name,rows_after=len(plays_25))

    # ── 2025-26: Real prop lines — ALL rows ───────────────────
    print("\n  Loading 2025-26 real prop lines (ALL rows, zero filtering)...")
    props = pd.read_excel(FILE_PROPS, sheet_name='Player_Points_Props', parse_dates=['Date'])
    print(f"  Real props: {len(props)}")

    print("  Running model on 2025-26...")
    plays_26 = run_model_on_props(props, all_logs, h2h, model, trust, '2025-26')
    graded_26 = [p for p in plays_26 if p['result'] in ('WIN','LOSS')]
    wins_26   = sum(1 for p in graded_26 if p['result']=='WIN')
    pct_26    = f"{wins_26/len(graded_26)*100:.1f}%" if graded_26 else "—"
    dnp_26    = sum(1 for p in plays_26 if p['result']=='DNP')
    print(f"  2025-26: {len(plays_26)} plays, {len(graded_26)} graded, {wins_26}W/{len(graded_26)-wins_26}L = {pct_26}, {dnp_26} DNP")

    with open(SEASON_2526,'w') as f: json.dump(clean_json(plays_26), f)
    print(f"  ✓ Saved {SEASON_2526.name} ({len(plays_26)} plays)")
    log_event('GEN','SEASON_2526_GENERATED',SEASON_2526.name,rows_after=len(plays_26))

    print(f"\n  Elapsed: {time.time()-t0:.1f}s")
    print("="*60)

if __name__ == '__main__': main()
