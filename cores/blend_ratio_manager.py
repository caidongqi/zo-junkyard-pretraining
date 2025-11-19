"""
Progressive Blend Ratio Manager
-------------------------------

This module implements a progressive scheduling strategy for the hybrid
blend ratio used in Instruct mode. The blend ratio controls how much of the
backprop (BP) gradient is mixed with the ZO gradient estimate. By default,
the scheduler keeps the initial ratio until the training loss goes below a
threshold, and then increases it in fixed increments for every additional
loss reduction step, up to a configured maximum.

The interface mirrors `InstructParamsManager` so it can be dropped into the
training loop with minimal boilerplate.
"""

# =====================================================================
# Configurable Hyperparameters (默认超参数，可按需覆盖)
# =====================================================================

# Loss threshold: start adjusting blend ratio when loss drops below this value
BLEND_RATIO_LOSS_THRESHOLD = 8.0

# Loss step: adjust blend ratio for every LOSS_STEP decrease in loss
BLEND_RATIO_LOSS_STEP = 0.2

# Blend ratio parameters
BLEND_RATIO_INITIAL = 0.0        # Initial blend ratio
BLEND_RATIO_INCREMENT = 0.2     # Increment per LOSS_STEP decrease
BLEND_RATIO_MAX = 0.95            # Maximum allowed blend ratio


# =====================================================================
# Blend Ratio Manager
# =====================================================================

class BlendRatioManager:
    """
    Progressive scheduler for the BP/ZO blend ratio.

    Usage:
        manager = BlendRatioManager(ratio_initial=0.1)
        ratio = manager.get_ratio(current_loss)
    """

    def __init__(
        self,
        loss_threshold: float = BLEND_RATIO_LOSS_THRESHOLD,
        loss_step: float = BLEND_RATIO_LOSS_STEP,
        ratio_initial: float = BLEND_RATIO_INITIAL,
        ratio_increment: float = BLEND_RATIO_INCREMENT,
        ratio_max: float = BLEND_RATIO_MAX,
    ):
        """
        Initialize the manager.

        Args:
            loss_threshold: Loss value below which adjustments begin.
            loss_step: Loss decrease required for one adjustment step.
            ratio_initial: Initial blend ratio (before threshold is reached).
            ratio_increment: Amount to increase ratio per step.
            ratio_max: Maximum allowable blend ratio.
        """
        if ratio_initial < 0.0 or ratio_initial > 1.0:
            raise ValueError("ratio_initial must be within [0, 1].")
        if ratio_max <= 0.0 or ratio_max > 1.0:
            raise ValueError("ratio_max must be within (0, 1].")
        if ratio_increment <= 0.0:
            raise ValueError("ratio_increment must be positive.")
        if loss_step <= 0.0:
            raise ValueError("loss_step must be positive.")

        ratio_max = max(ratio_max, ratio_initial)

        self.loss_threshold = loss_threshold
        self.loss_step = loss_step
        self.ratio_initial = max(0.0, min(ratio_initial, ratio_max))
        self.ratio_increment = ratio_increment
        self.ratio_max = ratio_max

        # Track the best (lowest) loss encountered so far
        self.best_loss = None

    def get_ratio(self, current_loss: float) -> float:
        """
        Calculate the scheduled blend ratio for the provided loss.

        Args:
            current_loss: Current training loss value.

        Returns:
            float: The scheduled blend ratio within [0, ratio_max].
        """
        if current_loss is None:
            return self.ratio_initial

        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss

        if self.best_loss >= self.loss_threshold:
            return self.ratio_initial

        loss_decrease = self.loss_threshold - self.best_loss
        num_steps = int(loss_decrease / self.loss_step)

        ratio = self.ratio_initial + num_steps * self.ratio_increment
        return min(ratio, self.ratio_max)

    def reset(self):
        """Reset the best loss tracker."""
        self.best_loss = None

    def get_current_state(self):
        """
        Return the current scheduling state for monitoring/logging.
        """
        if self.best_loss is None:
            return {
                "best_loss": None,
                "blend_ratio": self.ratio_initial,
                "status": "not_started",
            }

        ratio = self.get_ratio(self.best_loss)
        if self.best_loss >= self.loss_threshold:
            status = "before_threshold"
        elif ratio >= self.ratio_max:
            status = "at_max"
        else:
            status = "increasing"

        return {
            "best_loss": self.best_loss,
            "blend_ratio": ratio,
            "status": status,
            "loss_threshold": self.loss_threshold,
            "ratio_max": self.ratio_max,
        }

    def print_status(self):
        """Pretty-print the current scheduling status."""
        state = self.get_current_state()
        print("=" * 70)
        print("Blend Ratio Manager Status")
        print("=" * 70)
        print(
            f"Best Loss: {state['best_loss']:.4f}"
            if state["best_loss"] is not None
            else "Best Loss: Not started"
        )
        print(f"Status: {state['status']}")
        print(f"Current Blend Ratio: {state['blend_ratio']:.4f}")

        if state["status"] == "before_threshold":
            print(f"\n⏳ Waiting for loss to drop below {self.loss_threshold:.2f}")
        elif state["status"] == "increasing":
            print("\n🔄 Blend ratio increasing progressively")
            print(f"   Will max out at {self.ratio_max:.2f}")
        elif state["status"] == "at_max":
            print("\n✅ Blend ratio has reached the configured maximum")

        print("=" * 70)


# =====================================================================
# Convenience helpers
# =====================================================================

def create_default_blend_ratio_manager():
    """Create a manager with the default hyperparameters."""
    return BlendRatioManager()


def get_ratio_for_loss(loss_value, manager=None):
    """
    Convenience helper that returns the blend ratio for the given loss.
    """
    if manager is None:
        manager = create_default_blend_ratio_manager()
    return manager.get_ratio(loss_value)


