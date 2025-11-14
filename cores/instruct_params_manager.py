"""
Dynamic Instruction Parameters Manager

This module manages the dynamic adjustment of instruction parameters 
(cosine_target and noise_scale) based on training loss values.

The adjustment follows a progressive rule:
- Before loss reaches LOSS_THRESHOLD: use initial values
- After loss < LOSS_THRESHOLD: for every LOSS_STEP decrease in loss,
  - cosine_target increases by TARGET_INCREMENT
  - noise_scale decreases by SCALE_DECREMENT
- Until reaching extreme values (TARGET_MAX and SCALE_MIN), then remain constant
"""

# =====================================================================
# Configurable Hyperparameters (可配置的超参数)
# =====================================================================

# Loss threshold: start adjusting parameters when loss drops below this value
LOSS_THRESHOLD = 7.0

# Loss step: adjust parameters for every LOSS_STEP decrease in loss
LOSS_STEP = 0.2

# Cosine target parameters
TARGET_INITIAL = 0.01      # Initial cosine target value
TARGET_INCREMENT = 0.1     # Increment per LOSS_STEP decrease
TARGET_MAX = 0.99         # Maximum cosine target value

# Noise scale parameters
SCALE_INITIAL = 10.0      # Initial noise scale value
SCALE_DECREMENT = 1.0     # Decrement per LOSS_STEP decrease
SCALE_MIN = 0.1           # Minimum noise scale value


# =====================================================================
# Parameter Manager Class
# =====================================================================

class InstructParamsManager:
    """
    Manages dynamic adjustment of instruction parameters based on loss.
    
    Usage:
        manager = InstructParamsManager()
        cosine_target, noise_scale = manager.get_params(current_loss)
    """
    
    def __init__(
        self,
        loss_threshold=LOSS_THRESHOLD,
        loss_step=LOSS_STEP,
        target_initial=None,  # 如果为None，使用默认值TARGET_INITIAL
        target_increment=TARGET_INCREMENT,
        target_max=TARGET_MAX,
        scale_initial=None,  # 如果为None，使用默认值SCALE_INITIAL
        scale_decrement=SCALE_DECREMENT,
        scale_min=SCALE_MIN,
    ):
        """
        Initialize the parameter manager.
        
        Args:
            loss_threshold: Loss value below which adjustments begin
            loss_step: Loss decrease required for one adjustment step
            target_initial: Initial cosine target value (if None, uses TARGET_INITIAL default)
            target_increment: Amount to increase target per step
            target_max: Maximum cosine target value
            scale_initial: Initial noise scale value (if None, uses SCALE_INITIAL default)
            scale_decrement: Amount to decrease scale per step
            scale_min: Minimum noise scale value
        """
        self.loss_threshold = loss_threshold
        self.loss_step = loss_step
        # 如果target_initial为None，使用默认值；否则使用传入的值
        self.target_initial = target_initial if target_initial is not None else TARGET_INITIAL
        self.target_increment = target_increment
        self.target_max = target_max
        # 如果scale_initial为None，使用默认值；否则使用传入的值
        self.scale_initial = scale_initial if scale_initial is not None else SCALE_INITIAL
        self.scale_decrement = scale_decrement
        self.scale_min = scale_min
        
        # Track the best (lowest) loss seen so far
        self.best_loss = None
        
    def get_params(self, current_loss):
        """
        Calculate and return the appropriate parameters for the current loss.
        
        Args:
            current_loss: Current training loss value
            
        Returns:
            tuple: (cosine_target, noise_scale)
        """
        # Update best loss
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
        
        # Before threshold: use initial values
        if self.best_loss >= self.loss_threshold:
            return self.target_initial, self.scale_initial
        
        # After threshold: calculate adjustment steps
        loss_decrease = self.loss_threshold - self.best_loss
        num_steps = int(loss_decrease / self.loss_step)
        
        # Calculate cosine target (increasing)
        cosine_target = self.target_initial + (num_steps * self.target_increment)
        cosine_target = min(cosine_target, self.target_max)  # Cap at maximum
        
        # Calculate noise scale (decreasing)
        noise_scale = self.scale_initial - (num_steps * self.scale_decrement)
        noise_scale = max(noise_scale, self.scale_min)  # Cap at minimum
        
        return cosine_target, noise_scale
    
    def reset(self):
        """Reset the best loss tracker."""
        self.best_loss = None
    
    def get_current_state(self):
        """
        Get the current state of the manager.
        
        Returns:
            dict: Current state including best_loss and parameters
        """
        if self.best_loss is None:
            return {
                'best_loss': None,
                'cosine_target': self.target_initial,
                'noise_scale': self.scale_initial,
                'status': 'not_started'
            }
        
        cosine_target, noise_scale = self.get_params(self.best_loss)
        
        # Determine status
        if self.best_loss >= self.loss_threshold:
            status = 'before_threshold'
        elif cosine_target >= self.target_max or noise_scale <= self.scale_min:
            status = 'at_extreme'
        else:
            status = 'adjusting'
        
        return {
            'best_loss': self.best_loss,
            'cosine_target': cosine_target,
            'noise_scale': noise_scale,
            'status': status,
            'loss_threshold': self.loss_threshold,
            'target_max': self.target_max,
            'scale_min': self.scale_min,
        }
    
    def print_status(self):
        """Print the current status of parameter adjustment."""
        state = self.get_current_state()
        
        print("=" * 70)
        print("Instruct Parameters Manager Status")
        print("=" * 70)
        print(f"Best Loss: {state['best_loss']:.4f}" if state['best_loss'] is not None else "Best Loss: Not started")
        print(f"Status: {state['status']}")
        print(f"Current Cosine Target: {state['cosine_target']:.4f}")
        print(f"Current Noise Scale: {state['noise_scale']:.4f}")
        
        if state['status'] == 'before_threshold':
            print(f"\n⏳ Waiting for loss to drop below {state['loss_threshold']:.1f}")
        elif state['status'] == 'adjusting':
            print(f"\n🔄 Adjusting parameters progressively")
            print(f"   Target will max at {state['target_max']:.2f}")
            print(f"   Scale will min at {state['scale_min']:.2f}")
        elif state['status'] == 'at_extreme':
            print(f"\n✅ Parameters have reached extreme values")
        
        print("=" * 70)


# =====================================================================
# Convenience Functions
# =====================================================================

def create_default_manager():
    """Create a manager with default hyperparameters."""
    return InstructParamsManager()


def get_params_for_loss(loss_value, manager=None):
    """
    Convenience function to get parameters for a given loss value.
    
    Args:
        loss_value: Current training loss
        manager: Optional existing manager instance (if None, creates new one)
        
    Returns:
        tuple: (cosine_target, noise_scale)
    """
    if manager is None:
        manager = create_default_manager()
    return manager.get_params(loss_value)


# =====================================================================
# Example Usage
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Instruct Parameters Manager - Example Usage")
    print("=" * 70 + "\n")
    
    # Create manager with default settings
    manager = InstructParamsManager()
    
    # Simulate loss progression
    test_losses = [8.0, 7.5, 7.0, 6.8, 6.6, 6.4, 6.2, 6.0, 5.8, 5.6, 5.4, 5.2, 5.0]
    
    print("Loss Progression and Parameter Adjustment:\n")
    print(f"{'Loss':>8} | {'Cosine Target':>14} | {'Noise Scale':>12} | {'Status':>15}")
    print("-" * 70)
    
    for loss in test_losses:
        cosine_target, noise_scale = manager.get_params(loss)
        state = manager.get_current_state()
        print(f"{loss:>8.2f} | {cosine_target:>14.4f} | {noise_scale:>12.2f} | {state['status']:>15}")
    
    print("\n")
    manager.print_status()

