"""
Reliability Module.
Contains the SixSigmaGuard class for Bayesian fault detection.
"""
import numpy as np


class SixSigmaGuard:
    """
    Implements a Bayesian Recursive Filter for Six Sigma reliability.
    Accumulates sensor evidence over time to rule out false positives.
    """

    def __init__(self):
        # Initial Belief (Log Odds)
        # 0.0 means 50/50 chance. -5.0 means "Almost certainly safe"
        self.log_odds = -5.0

        # Sensor Characteristics (Calibrated Lab Data)
        # True Positive Probability (Sensitivity)
        self.p_gas_given_leak = 0.95
        self.p_temp_given_leak = 0.90

        # False Positive Probability (1 - Specificity)
        self.p_gas_given_safe = 0.05
        self.p_temp_given_safe = 0.10

        # Thresholds
        self.sigma_6_threshold = 15.0  # Log odds equivalent to 99.9999%

    def update(self, gas_reading, temp_reading, dt):
        """
        Bayesian Update Step.
        Runs at 100Hz. Accumulates evidence over time.

        Args:
            gas_reading: Current gas sensor value.
            temp_reading: Current thermal sensor value.
            dt: Time delta in seconds.

        Returns:
            (is_confirmed, confidence_score)
        """
        # 1. Calculate Likelihood of Evidence
        # GAS SENSOR MODEL
        if gas_reading > 50.0:
            # Evidence supports leak
            l_gas = np.log(self.p_gas_given_leak / self.p_gas_given_safe)
        else:
            # Evidence supports safe
            l_gas = np.log((1 - self.p_gas_given_leak) /
                           (1 - self.p_gas_given_safe))

        # THERMAL SENSOR MODEL (Joule-Thomson Cooling)
        if temp_reading < 26.0:
            l_temp = np.log(self.p_temp_given_leak / self.p_temp_given_safe)
        else:
            l_temp = np.log((1 - self.p_temp_given_leak) /
                            (1 - self.p_temp_given_safe))

        # 2. Recursive Update (Bayes Rule in Log Domain)
        # New Belief = Old Belief + Evidence + Time_Decay
        decay = -0.1 * dt  # Belief decays if evidence stops (Auto-Reset)
        self.log_odds = self.log_odds + l_gas + l_temp + decay

        # Clamp to prevent overflow
        self.log_odds = np.clip(self.log_odds, -20.0, 20.0)

        # 3. Check Sigma Level
        confidence = 1.0 / (1.0 + np.exp(-self.log_odds))
        is_confirmed = self.log_odds > self.sigma_6_threshold

        return is_confirmed, confidence
