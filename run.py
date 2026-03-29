#!/usr/bin/env python3
"""
PropEdge V9.2 — Master Runner
  python3 run.py 0           # Batch 0: grade (6 AM UK)
  python3 run.py 1           # Batch 1: early lines (8 AM UK)
  python3 run.py 2           # Batch 2: main run (6 PM UK)
  python3 run.py 3           # Batch 3: pre-tip
  python3 run.py all         # Batch 0 then 2
  python3 run.py generate    # One-time: regenerate season JSONs
  python3 run.py setup       # Git init + SSH remote + launchd agents
"""
import sys, subprocess, os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
from config import GIT_REMOTE, REPO_DIR, VERSION


def run_batch(n):
    if n == 0:
        subprocess.run([sys.executable, str(ROOT / 'batch0_grade.py')], cwd=ROOT)
    else:
        subprocess.run([sys.executable, str(ROOT / 'batch_predict.py'), str(n)], cwd=ROOT)


def setup():
    print(f"\n  PropEdge {VERSION} — Setup")

    # ── Git init ──
    if not (ROOT / '.git').exists():
        subprocess.run(['git','init'], cwd=ROOT)
        subprocess.run(['git','add','-A'], cwd=ROOT)
        subprocess.run(['git','commit','-m',f'Initial commit {VERSION}'], cwd=ROOT)
        print("  ✓ Git repo initialised")
    else:
        print("  ✓ Git repo exists")

    # ── Set SSH remote ──
    result = subprocess.run(['git','remote','get-url','origin'],
                            cwd=ROOT, capture_output=True, text=True)
    current = result.stdout.strip() if result.returncode == 0 else ''
    if current != GIT_REMOTE:
        if current:
            subprocess.run(['git','remote','remove','origin'], cwd=ROOT, capture_output=True)
            print(f"  Removed old remote: {current}")
        subprocess.run(['git','remote','add','origin',GIT_REMOTE], cwd=ROOT, capture_output=True)
        print(f"  ✓ Remote set (SSH): {GIT_REMOTE}")
    else:
        print(f"  ✓ Remote OK: {GIT_REMOTE}")

    # ── Test SSH ──
    try:
        r = subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=no',
                            'git@github.com'], capture_output=True, text=True, timeout=15)
        if 'successfully authenticated' in r.stderr.lower():
            print("  ✓ SSH authentication confirmed")
        else:
            print(f"  ⚠ SSH: {r.stderr.strip()[:100]}")
            print("    Run: ssh-add ~/.ssh/id_ed25519")
    except Exception as e:
        print(f"  ⚠ SSH test failed: {e}")

    # ── Initial push ──
    env = {**os.environ, 'GIT_SSH_COMMAND':'ssh -o BatchMode=yes'}
    try:
        subprocess.run(['git','push','-u','origin','main'], cwd=ROOT,
                       capture_output=True, timeout=60, env=env)
        print("  ✓ Initial push complete")
    except Exception as e:
        print(f"  ⚠ Initial push: {e}")

    # ── Batch 3 dynamic wrapper ──
    py    = sys.executable
    home  = Path.home()
    pdir  = home / 'Library' / 'LaunchAgents'
    pdir.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)

    wrapper = ROOT / 'batch3_dynamic.sh'
    wrapper.write_text(f"""#!/bin/bash
cd {ROOT}
FIRST_GAME=$({py} -c "
import requests,json
from datetime import datetime,timezone,timedelta
try:
    r=requests.get('https://api.the-odds-api.com/v4/sports/basketball_nba/events',
      params={{'apiKey':'c0bab20a574208a41a6e0d930cdaf313','dateFormat':'iso'}},timeout=15)
    events=r.json()
    today=datetime.now(timezone(timedelta(hours=-4))).strftime('%Y-%m-%d')
    times=[datetime.fromisoformat(e['commence_time'].replace('Z','+00:00'))
           for e in events
           if datetime.fromisoformat(e['commence_time'].replace('Z','+00:00'))
                     .astimezone(timezone(timedelta(hours=-4))).strftime('%Y-%m-%d')==today]
    if times: print(min(times).astimezone(timezone(timedelta(hours=1))).strftime('%H:%M'))
    else: print('22:00')
except: print('22:00')
" 2>/dev/null)

HOUR=${{FIRST_GAME%%:*}}
TARGET_HOUR=$((HOUR - 1))
CURRENT_HOUR=$(date +%H)
if [ "$CURRENT_HOUR" -ge "$TARGET_HOUR" ]; then
    {py} {ROOT}/batch_predict.py 3
fi
""")
    os.chmod(wrapper, 0o755)

    # ── Launchd agents ──
    schedules = [
        ('grade',  'batch0_grade.py',   '',  6,  0),
        ('batch1', 'batch_predict.py',  '1', 8,  0),
        ('batch2', 'batch_predict.py',  '2', 18, 0),
        ('batch3', 'batch3_dynamic.sh', '',  21, 30),
    ]

    for label, script, args, hour, minute in schedules:
        pn = f'com.propedge.v92.{label}'
        if script.endswith('.sh'):
            pa = [str(ROOT / script)]
        else:
            pa = [py, str(ROOT / script)]
            if args: pa.append(args)

        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{pn}</string>
  <key>ProgramArguments</key>
  <array>{''.join(f'<string>{a}</string>' for a in pa)}</array>
  <key>WorkingDirectory</key><string>{ROOT}</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>{hour}</integer>
    <key>Minute</key><integer>{minute}</integer>
  </dict>
  <key>StandardOutPath</key><string>{ROOT}/logs/{label}.log</string>
  <key>StandardErrorPath</key><string>{ROOT}/logs/{label}_err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    <key>HOME</key><string>{home}</string>
  </dict>
</dict>
</plist>"""

        pf = pdir / f'{pn}.plist'
        pf.write_text(plist)
        subprocess.run(['launchctl','unload', str(pf)], capture_output=True)
        subprocess.run(['launchctl','load',   str(pf)], capture_output=True)
        print(f"  ✓ {pn} → {hour:02d}:{minute:02d} UK")

    print(f"\n  Logs: {ROOT}/logs/")
    print(f"  Setup complete. Remote: {GIT_REMOTE}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == 'setup':
        setup()
    elif cmd == 'generate':
        subprocess.run([sys.executable, str(ROOT / 'generate_season_json.py')], cwd=ROOT)
    elif cmd == 'all':
        run_batch(0); run_batch(2)
    elif cmd in ('0','1','2','3'):
        run_batch(int(cmd))
    else:
        print(f"Unknown command: {cmd}"); print(__doc__)
