#!/usr/bin/env python3
"""
ORB_Bot - Heartbeat Runner
Wird beim US-Marktstart aufgerufen (9:30 AM ET = 15:30 CEST).
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path

def run_orb_bot():
    """Führt den ORB_Bot aus"""
    print("[ORB Heartbeat] Starte ORB_Bot...")
    
    try:
        result = subprocess.run(
            ["python3", "/data/.openclaw/workspace/orb_bot.py"],
            capture_output=True,
            text=True,
            cwd="/data/.openclaw/workspace",
            timeout=120
        )
        
        # Ergebnis loggen
        log_file = Path("/data/.openclaw/workspace/orb_trading_data/orb_heartbeat.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Run: {datetime.now()}\n")
            f.write(f"{'='*60}\n")
            f.write(result.stdout)
            if result.stderr:
                f.write("\n[ERRORS]\n")
                f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"[ORB Heartbeat] Erfolgreich abgeschlossen: {datetime.now()}")
            return True
        else:
            print(f"[ORB Heartbeat] Fehler beim Ausführen: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"[ORB Heartbeat] Ausnahme: {e}")
        return False

def should_run():
    """Prüfen ob wir innerhalb der US-Handelszeit sind und alle 30 Minuten laufen sollen"""
    now = datetime.now()
    
    # US-Handelszeit: 9:30 ET - 16:00 ET = 15:30 CEST - 22:00 CEST (UTC+2)
    start_hour, start_minute = 15, 30  # 15:30 CEST
    end_hour, end_minute = 22, 0       # 22:00 CEST
    
    current_time = now.hour * 60 + now.minute
    start_time = start_hour * 60 + start_minute
    end_time = end_hour * 60 + end_minute
    
    # Prüfen ob wir im Handelszeitfenster sind
    if not (start_time <= current_time < end_time):
        return False
    
    # Alle 15 Minuten laufen (Minute 00, 15, 30, 45)
    return now.minute in [0, 15, 30, 45]

def main():
    if should_run():
        success = run_orb_bot()
        sys.exit(0 if success else 1)
    else:
        # Nur für Debugging - normalerweise nichts ausgeben
        # print(f"[ORB Heartbeat] Nicht die richtige Zeit. Aktuell: {datetime.now().strftime('%H:%M')}")
        sys.exit(0)

if __name__ == "__main__":
    main()