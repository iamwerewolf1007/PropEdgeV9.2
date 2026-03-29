# PropEdge V9.2 — NBA Player Points Prop Prediction

## What's new in V9.2
1. **Rolling stats cross-validation** — At grade time, L30 is recomputed fresh from the game log CSV and compared against the value stored in the play JSON. Deviations >1pt are flagged to `audit_log.csv` and appended to `postMatchReason`.
2. **Live rolling** — All rolling stats computed on-demand from played games only. No stale CSV snapshot columns used at prediction time.
3. **DNP handling** — Stub rows appended to game log CSV with `DNP=1`. Excluded from all rolling calculations, hit rates, and leaderboard stats.
4. **Post-match reasoning** — Same analytical depth as pre-match. 7-part structure using actual box score data cross-referenced against prediction-time stored values.
5. **Dashboard** — Date navigator, search fix (Enter/clear only), tier filter boxes in daily stats bar, DNP badge, L20 high-contrast sparklines.

## Batches
| Time (UK) | Script | Description |
|-----------|--------|-------------|
| 06:00 | batch0_grade.py | Grade + append game logs + retrain |
| 08:00 | batch_predict.py 1 | Early props |
| 18:00 | batch_predict.py 2 | Main prediction run |
| 21:30 | batch3_dynamic.sh | Pre-tip |

## Setup
```bash
cd ~/Documents/GitHub/PropEdgeV9.2
python3 run.py setup      # Git + SSH + launchd
python3 run.py generate   # Build both season JSONs
python3 run.py 2          # Test run
```
