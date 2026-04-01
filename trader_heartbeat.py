#!/usr/bin/env python3
"""
Botti Trader - Heartbeat Runner
Wird bei jedem Heartbeat aufgerufen.
"""

import sys
sys.path.insert(0, '/data/.openclaw/workspace')

from trader_daemon import run_trader

def check_and_run():
    """Führt den Trader aus"""
    print("[Heartbeat] Starte Botti Trader...")
    success = run_trader()
    return success

if __name__ == "__main__":
    result = check_and_run()
    sys.exit(0 if result in [True, None] else 1)
