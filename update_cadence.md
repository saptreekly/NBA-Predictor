# NBA Predictor - Update Cadence Timetable

This document outlines the schedule, triggers, and service level agreement for refreshing and publishing NBA game predictions.

---

## 1. Cadence Tiers & Cron Times

### Morning Batch
*   **Schedule:** Daily at 06:00 AM ET
*   **UTC Equivalent:** 10:00 AM UTC
*   **NZT Equivalent:** 10:00 PM NZT (prior day)
*   **Cron String (UTC):** `0 10 * * *`
*   **Description:** A comprehensive daily prediction run covering all upcoming games.

### Pre-tip Refresh
*   **Schedule:** Every 15 minutes for games within 120 minutes (2 hours) of their scheduled tip-off.
*   **Cron String (UTC):** `*/15 * * * *` (The underlying script will contain logic to identify and process only relevant games within the 2-hour window).
*   **Description:** Frequent updates for games approaching their start time, incorporating the latest available information.

---

## 2. Refresh Triggers

Predictions will be re-evaluated and refreshed based on the following significant events:
*   **Injury Status Changes:** Key player injury updates or changes in availability.
*   **Line Movement:** Substantial shifts in betting lines (e.g., spread, total) from major sportsbooks.
*   **Game Start Time Changes:** Official adjustments to a game's scheduled tip-off time.
*   **Major News:** Any significant news or developments impacting team rosters or performance (e.g., coaching changes, trades).

---

## 3. Cutoff Rules

*   **Pregame Prediction Lock:** All pregame predictions will be locked and will cease to refresh once the game's scheduled tip-off time is reached. This ensures a clear audit trail for pregame prediction performance.

---

## 4. Service Level Agreement (SLA) Text

"Pregame predictions for all scheduled NBA games will be available at least 2 hours before the earliest game's scheduled tip-off time each day. Subsequent pre-tip refreshes will occur every 15 minutes for games within a 2-hour window of their scheduled start."
