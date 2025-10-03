# NBA Predictor - Prediction Types

This document outlines the selected prediction types for the NBA predictor, including their purpose, numerical characteristics, and publishing status.

---

## 1. Pregame Home Win Probability

*   **Purpose + User:** This metric quantifies the likelihood of the home team securing a victory before the game commences. It serves as a quick reference for bettors, fans, and analysts to assess the favored team.
*   **Numerical Range & Units:** [0, 1], unitless (representing a probability).
*   **Sign Convention:** N/A (a higher value indicates a greater probability of the home team winning).
*   **Rounding/Precision:** 3 decimal places (e.g., 0.755).
*   **Publishing Status:** `beta`

---

## 2. Pregame Predicted Margin (Home - Away)

*   **Purpose + User:** This metric estimates the expected point differential between the home and away teams. It helps users understand the anticipated dominance of one team and is particularly useful for spread betting.
*   **Numerical Range & Units:** Real-valued number, in points.
*   **Sign Convention:** A positive value signifies that the home team is predicted to win by that margin, while a negative value indicates the away team is predicted to win by that margin.
*   **Rounding/Precision:** 1 decimal place (e.g., +7.5 points).
*   **Publishing Status:** `beta`

---

## 3. Pregame Predicted Total (Sum of Points)

*   **Purpose + User:** This metric forecasts the combined total score of both teams in the game. It is essential for "over/under" betting markets and provides insight into the expected scoring environment and pace of play.
*   **Numerical Range & Units:** Positive real-valued number, in points.
*   **Sign Convention:** N/A.
*   **Rounding/Precision:** 1 decimal place (e.g., 225.5 points).
*   **Publishing Status:** `beta`
