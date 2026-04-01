---
summary: "Daily trading bot + memory manager checks"
read_when:
  - Heartbeat polling
---

# HEARTBEAT.md - Botti Trader + ORB_Bot + Memory Manager Tasks

## Anweisung fuer Heartbeats

Bei jedem Heartbeat:
1. Pruefe die aktuelle Zeit
2. Fuehre aus: `python3 /data/.openclaw.workspace/trader_heartbeat.py` (stündlich Report senden)
3. Fuehre aus: `python3 /data/.openclaw.workspace/orb_heartbeat.py` (US-Marktstart um 15:30 CEST)
4. Alle 2 Stunden: `bash /data/.openclaw.workspace/skills/memory-manager/detect.sh` (Memory Check)
5. Wenn WARN/CRITICAL: `bash /data/.openclaw.workspace/skills/memory-manager/snapshot.sh`
6. Täglich 23:00: `bash /data/.openclaw.workspace/skills/memory-manager/organize.sh`

## Trading Zeiten
- **Botti Trader (Original):** 08:00-22:00 - Stündlicher Scan + Report senden (auch wenn HOLD)
- **ORB_Bot (Neu):** Täglich um 15:30 CEST (US-Marktstart 9:30 AM ET)
- **Memory Organizer:** Täglich 23:00

## Letzte Laeufe
check memory/heartbeat-state.json fuer "lastRuns"

## Memory Manager
- Alle 2h: Compression Check
- Bei >70%: WARNING
- Bei >85%: CRITICAL + Snapshot
- 23:00: Automatische Organisation