#!/bin/bash
cd /Users/salman/Documents/GitHub/PropEdgeV9.2
FIRST_GAME=$(/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -c "
import requests,json
from datetime import datetime,timezone,timedelta
try:
    r=requests.get('https://api.the-odds-api.com/v4/sports/basketball_nba/events',
      params={'apiKey':'c0bab20a574208a41a6e0d930cdaf313','dateFormat':'iso'},timeout=15)
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

HOUR=${FIRST_GAME%%:*}
TARGET_HOUR=$((HOUR - 1))
CURRENT_HOUR=$(date +%H)
if [ "$CURRENT_HOUR" -ge "$TARGET_HOUR" ]; then
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 /Users/salman/Documents/GitHub/PropEdgeV9.2/batch_predict.py 3
fi
