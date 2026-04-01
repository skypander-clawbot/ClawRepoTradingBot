#!/usr/bin/env python3
"""
Botti Trader Daemon - Einfacher Scheduler
=======================================
Speichert den nächsten geplanten Lauf und führt den Trader aus.
"""

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

CONFIG = {
    "schedule_file": Path("/data/.openclaw/workspace/trading_data/schedule.json"),
    "heartbeat_file": Path("/data/.openclaw/workspace/memory/heartbeat-state.json"),
    "run_times": ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"],  # 9 Uhr und 21 Uhr
}

def load_schedule():
    """Gespeicherten Schedule laden"""
    if CONFIG["schedule_file"].exists():
        with open(CONFIG["schedule_file"], "r") as f:
            return json.load(f)
    return {"last_run": None, "next_runs": []}

def save_schedule(schedule):
    """Schedule speichern"""
    CONFIG["schedule_file"].parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["schedule_file"], "w") as f:
        json.dump(schedule, f, indent=2)

def update_heartbeat(timestamp=None):
    """Heartbeat-Status aktualisieren"""
    try:
        if CONFIG["heartbeat_file"].exists():
            with open(CONFIG["heartbeat_file"], "r") as f:
                state = json.load(f)
        else:
            state = {"trader": {"lastRuns": {}}}
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Determine which hour slot we are in (based on run_times)
        now = datetime.now()
        current_hour = now.hour
        # Find the run_time that matches current hour (assuming minute 0)
        matched_rt = None
        for rt in CONFIG["run_times"]:
            rt_hour = int(rt.split(":")[0])
            if rt_hour == current_hour:
                matched_rt = rt
                break
        if matched_rt:
            state["trader"]["lastRuns"][matched_rt] = timestamp
        
        with open(CONFIG["heartbeat_file"], "w") as f:
            json.dump(state, f, indent=2)
            
    except Exception as e:
        print(f"[Heartbeat] Update fehlgeschlagen: {e}")

def get_last_run(time_slot):
    """Letzten Lauf fuer Zeit-Slot abrufen"""
    try:
        if CONFIG["heartbeat_file"].exists():
            with open(CONFIG["heartbeat_file"], "r") as f:
                state = json.load(f)
                return state.get("trader", {}).get("lastRuns", {}).get(time_slot)
    except:
        pass
    return None

def should_run():
    """Pruefen ob jetzt ein Lauf ansteht (und heute noch nicht gelaufen)"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    today = now.strftime("%Y-%m-%d")
    
    # Pruefe ob wir in der richtigen Minute sind
    for rt in CONFIG["run_times"]:
        rt_hour, rt_min = map(int, rt.split(":"))
        if now.hour == rt_hour and now.minute == rt_min:
            # Pruefe ob heute schon gelaufen
            last_run = get_last_run(rt)
            if last_run:
                last_run_date = last_run[:10]  # YYYY-MM-DD
                if last_run_date == today:
                    return False  # Heute schon gelaufen
            return True
    
    return False

def run_trader():
    """Trader ausführen"""
    print(f"[Trader] Starte Analyse: {datetime.now()}")
    
    try:
        result = subprocess.run(
            ["python3", "/data/.openclaw/workspace/trader_v5.py"],
            capture_output=True,
            text=True,
            cwd="/data/.openclaw/workspace",
            timeout=120
        )
        
        # Ergebnis loggen
        log_file = Path("/data/.openclaw/workspace/trading_data/daemon.log")
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Run: {datetime.now()}\n")
            f.write(f"{'='*60}\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n[ERRORS]\n")
                f.write(result.stderr)
        
        if result.returncode == 0:
            update_heartbeat()  # Heartbeat aktualisieren
        
        print(f"[Trader] Fertig: {datetime.now()}")
        return True
        
    except Exception as e:
        print(f"[Trader] Fehler: {e}")
        return False

def get_next_runs():
    """Nächste geplante Läufe berechnen"""
    now = datetime.now()
    today = now.date()
    runs = []
    
    for time_str in CONFIG["run_times"]:
        hour, minute = map(int, time_str.split(":"))
        run_time = datetime.combine(today, __import__('datetime').time(hour, minute))
        
        if run_time > now:
            runs.append(run_time)
    
    # Wenn keine heute mehr, nimm morgen
    if not runs:
        for time_str in CONFIG["run_times"]:
            hour, minute = map(int, time_str.split(":"))
            run_time = datetime.combine(today + timedelta(days=1), __import__('datetime').time(hour, minute))
            runs.append(run_time)
    
    return sorted(runs)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "run":
            # Sofort ausführen
            run_trader()
        elif sys.argv[1] == "status":
            # Status anzeigen
            schedule = load_schedule()
            next_runs = get_next_runs()
            print("Botti Trader Status")
            print(f"Letzter Lauf: {schedule.get('last_run', 'Nie')}")
            print(f"Nächste Läufe:")
            for run in next_runs[:3]:
                print(f"  - {run.strftime('%Y-%m-%d %H:%M')}")
        elif sys.argv[1] == "schedule":
            # Cron-ähnlicher Output
            next_runs = get_next_runs()
            for run in next_runs:
                print(run.isoformat())
    else:
        # Prüfen ob jetzt ausgeführt werden soll
        if should_run():
            run_trader()
        else:
            next_runs = get_next_runs()
            print(f"Nichts zu tun. Nächster Lauf: {next_runs[0].strftime('%Y-%m-%d %H:%M')}")
