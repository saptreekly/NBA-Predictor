# Uncertainty Policy for NBA Prediction Models

## 1. Current Approach (v1 - Point Estimates Only)

At this initial stage, our NBA prediction models (Win Probability, Margin, Total) will publish only point estimates. This approach prioritizes simplicity and immediate utility, with the understanding that robust uncertainty quantification will be integrated in future iterations.

*   **Win Probability:** The model will output a single probability value (e.g., 0.753 for a home win).
*   **Margin:** The model will output a single predicted point differential (e.g., 5.2 points).
*   **Total:** The model will output a single predicted total score (e.g., 210.5 points).

## 2. Future Uncertainty Quantification Strategy

We are committed to providing well-calibrated uncertainty estimates to enhance the reliability and interpretability of our predictions. Our future strategy will involve:

*   **Win Probability:** We plan to implement empirical quantiles, potentially via ensembling or post-hoc calibration techniques (e.g., Isotonic Regression) applied to `predict_proba` outputs. This will allow us to generate credible intervals.
*   **Margin & Total:** We plan to utilize bootstrap resampling methods to generate prediction intervals around our linear regression point estimates. This will provide empirical confidence bands for the predicted margin and total scores.

## 3. Coverage Targets (Future)

Once uncertainty intervals are implemented, we will target the following coverage bands:

*   **80% Prediction Interval:** This band will aim to contain the true outcome 80% of the time.
*   **95% Prediction Interval:** This band will aim to contain the true outcome 95% of the time.

## 4. Calibration Evaluation (Future)

Rigorous evaluation of our uncertainty estimates is paramount.

*   **Win Probability:** Calibration will be assessed using **reliability curves** (also known as calibration plots). These plots compare the predicted probabilities to the observed frequencies of outcomes, allowing us to identify and correct for miscalibration.
*   **Margin & Total:** Calibration of prediction intervals will be evaluated by tracking their **empirical coverage**. We will assess whether the true outcomes fall within the 80% and 95% prediction intervals at the expected rates.

## 5. Output Fields (Future)

When intervals are introduced, the output will be expanded to include fields such as:

*   `win_prob_p10`, `win_prob_p50`, `win_prob_p90` (for win probability quantiles)
*   `margin_lower_80`, `margin_upper_80`, `margin_lower_95`, `margin_upper_95` (for margin prediction intervals)
*   `total_lower_80`, `total_upper_80`, `total_lower_95`, `total_upper_95` (for total prediction intervals)

## 6. Rounding Rules

To maintain consistency and readability, the following rounding rules will be applied to our published predictions:

*   **Win Probability:** Rounded to 3 decimal places (e.g., 0.753).
*   **Margin:** Rounded to 1 decimal place (e.g., 5.2 points).
*   **Total:** Rounded to 1 decimal place (e.g., 210.5 points).

## Example of a Calibrated Band (Conceptual - Future State)

*   **Win Probability:** A prediction of "Home Team Win Probability: 0.75" might eventually be accompanied by an 80% credible interval of `[0.68, 0.82]`. This would imply that, based on our calibrated model, there is an 80% chance the true home win probability lies between 68% and 82%.
*   **Margin:** A prediction of "Predicted Margin: Home by 5.2 points" might eventually be accompanied by a 95% prediction interval of `[-2.5, 13.0]`. This would imply that, based on our calibrated model, there is a 95% chance the actual game margin will fall between a 2.5-point away win and a 13.0-point home win.

---