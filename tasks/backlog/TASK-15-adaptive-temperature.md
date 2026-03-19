# TASK-15: Adaptive Temperature Scheduling

- **Category:** training-dynamics
- **Priority:** 8.5/10
- **Impact:** 7/10
- **Feasibility:** 9/10
- **Confidence:** 8/10

## Problem Statement

**Current fixed schedule issues:**
- Cosine 1.0 → 0.7 either learns too slowly or plateaus
- One-size-fits-all doesn't adapt to model state
- No feedback from validation performance
- Temperature affects different models differently

**Root cause:** Temperature controls exploration/exploitation tradeoff, but optimal schedule depends on:
- Current routing entropy (are decisions sharp enough?)
- Validation accuracy trend (are we improving or stuck?)
- Model capacity (bigger models need different schedules)

## Proposed Solution

### **Strategy A: Validation-Responsive Temperature**

Adjust temperature based on validation accuracy improvement:

```python
class AdaptiveTemperatureScheduler:
    def __init__(self, initial=1.0, patience=3, decay_factor=0.95):
        self.temp = initial
        self.patience = patience
        self.decay_factor = decay_factor
        self.best_val_acc = 0.0
        self.steps_since_improvement = 0
    
    def step(self, val_acc):
        if val_acc > self.best_val_acc + 0.001:  # 0.1% threshold
            # Improving - keep temperature
            self.best_val_acc = val_acc
            self.steps_since_improvement = 0
        else:
            # Plateauing - sharpen decisions
            self.steps_since_improvement += 1
            if self.steps_since_improvement >= self.patience:
                self.temp *= self.decay_factor
                self.temp = max(self.temp, 0.5)  # floor at 0.5
                self.steps_since_improvement = 0
        return self.temp
```

### **Strategy B: Entropy-Target Temperature Controller**

PID-style controller targeting desired entropy:

```python
class EntropyTargetScheduler:
    def __init__(self, target_entropy=0.3, Kp=0.1, Ki=0.01):
        self.target = target_entropy
        self.Kp = Kp
        self.Ki = Ki
        self.integral = 0.0
        self.temp = 1.0
    
    def step(self, current_entropy):
        error = current_entropy - self.target
        self.integral += error
        # P + I control (no D to avoid noise)
        adjustment = self.Kp * error + self.Ki * self.integral
        self.temp = np.clip(self.temp - adjustment, 0.5, 1.5)
        return self.temp
```

**Target entropy guide:**
- 0.69 = uniform (completely soft)
- 0.3-0.4 = good exploration/exploitation balance
- <0.1 = nearly deterministic

### **Strategy C: Hybrid Approach**

Combine validation trend + entropy target:

```python
def hybrid_temperature(val_acc, entropy, step, total_steps):
    # Base: Cosine schedule
    progress = step / total_steps
    cosine_temp = 0.7 + 0.3 * (1 + np.cos(np.pi * progress)) / 2
    
    # Modifier 1: If plateauing, sharpen faster
    if val_acc_plateau_detected():
        cosine_temp *= 0.95
    
    # Modifier 2: Entropy bounds
    if entropy > 0.5:  # too soft
        cosine_temp = min(cosine_temp, 0.8)
    elif entropy < 0.2:  # too sharp
        cosine_temp = max(cosine_temp, 0.9)
    
    return np.clip(cosine_temp, 0.5, 1.0)
```

## Implementation

**Files to modify:**
- `train.py` - Add temperature scheduler class before training loop
- `train.py` - Replace fixed cosine with adaptive scheduler
- `main.py` - Ensure `set_temperature()` is called every eval interval

**New parameters:**
```python
parser.add_argument('--temp_schedule', type=str, 
                    choices=['cosine', 'adaptive', 'entropy_target', 'hybrid'],
                    default='adaptive')
parser.add_argument('--temp_patience', type=int, default=3,
                    help='Eval intervals to wait before lowering temp')
parser.add_argument('--target_entropy', type=float, default=0.3)
```

## Expected Impact

- **Accuracy:** +0.5-1.5pp by avoiding premature convergence
- **Training stability:** Better convergence across different configs
- **Robustness:** Less sensitive to hyperparameter choices

## Experiment Plan

Compare 4 schedules on same model (oblivious_boosted_vo_alt):
1. Fixed cosine 1.0 → 0.7 (baseline)
2. Validation-responsive (Strategy A)
3. Entropy-target (Strategy B)
4. Hybrid (Strategy C)

Track: val_acc, entropy, temperature over training.

## Risks

- PID tuning might need adjustment per model size
- Could oscillate if val_acc is noisy (add smoothing)
- May need different schedules for different tree types

## Dependencies

None. Can implement immediately.
