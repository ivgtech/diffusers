# %% 

import jax
import jax.numpy as jnp
import unittest

def get_decay1(
    step: int,
    max_ema_decay: float = 0.9999,
    min_ema_decay: float = 0.0,
    ema_inv_gamma: float = 1.0,
    ema_decay_power: float = 2 / 3,
    use_ema_warmup: bool = False,
    start_ema_update_after_n_steps: float = 10.0  # Mimic diffusers default value
):
    # Adjust step to consider the start update offset
    adjusted_step = jnp.maximum(step - start_ema_update_after_n_steps - 1, 0)
    
    # Compute base decay depending on the warmup usage
    if use_ema_warmup:
        decay = 1.0 - (1.0 + adjusted_step / ema_inv_gamma) ** -ema_decay_power
    else:
        decay = (1.0 + adjusted_step) / (10.0 + adjusted_step) if start_ema_update_after_n_steps == 0 \
                else (1.0 + adjusted_step) / (start_ema_update_after_n_steps + adjusted_step)

    # Scale the decay by a multiple which is zero before the start and one afterwards
    multiple = jnp.where(step > start_ema_update_after_n_steps, 1.0, 0.0)
    decay *= multiple

    # Clip the decay to ensure it stays within the specified bounds
    return jnp.clip(decay, min_ema_decay, max_ema_decay)

def get_decay2(
    step: int, max_ema_decay: float = 0.9999,
    min_ema_decay: float = 0.0,
    ema_inv_gamma: float = 1.0,
    ema_decay_power: float = 2 / 3, 
    use_ema_warmup: bool = False,
    start_ema_update_after_n_steps: float = 10.0
    ):
    # Adjust step to consider the start update offset
    adjusted_step = jnp.maximum(step - start_ema_update_after_n_steps, 0)

    # Compute base decay
    if use_ema_warmup:
        decay = 1.0 - (1.0 + adjusted_step / ema_inv_gamma) ** -ema_decay_power
    else:
        initial_steps = jnp.where(start_ema_update_after_n_steps == 0, 10.0, start_ema_update_after_n_steps)
        decay = (1.0 + adjusted_step) / (initial_steps + adjusted_step)

    # Ensure decay starts changing only after certain steps
    decay = jnp.where(step > start_ema_update_after_n_steps, decay, min_ema_decay)

    # Clip the decay to ensure it stays within the specified bounds
    return jnp.clip(decay, min_ema_decay, max_ema_decay)

# Reference decay function from training_utils.py
def reference_decay(step, max_ema_decay, min_ema_decay, ema_inv_gamma, ema_decay_power, use_ema_warmup, start_ema_update_after_n_steps):
    adjusted_step = max(step - start_ema_update_after_n_steps, 0)
    if use_ema_warmup:
        decay = 1.0 - (1.0 + adjusted_step / ema_inv_gamma) ** -ema_decay_power
    else:
        initial_steps = max(start_ema_update_after_n_steps, 10.0)
        decay = (1.0 + adjusted_step) / (initial_steps + adjusted_step)
    decay = max(min(decay, max_ema_decay), min_ema_decay)
    return decay

# Test parameters
steps = [0, 5, 10, 50, 100]
max_ema_decay = 0.999
min_ema_decay = 0.0
ema_inv_gamma = 1.0
ema_decay_power = 2 / 3
use_ema_warmup = True 
start_ema_update_after_n_steps = 100.0
step = 1000

decay1 = get_decay1(
    step,
    max_ema_decay,
    min_ema_decay,
    ema_inv_gamma,
    ema_decay_power,
    use_ema_warmup,
    start_ema_update_after_n_steps
)
decay2 = get_decay2(
    step,
    max_ema_decay,
    min_ema_decay,
    ema_inv_gamma,
    ema_decay_power,
    use_ema_warmup,
    start_ema_update_after_n_steps
)

expected_decay = reference_decay(
    step,
    max_ema_decay,
    min_ema_decay,
    ema_inv_gamma,
    ema_decay_power,
    use_ema_warmup,
    start_ema_update_after_n_steps
)

print(f"Decay1: {decay1}",
      f"\nDecay2: {decay2}",
      f"\nExpected Decay: {expected_decay}"
      )
# %% 
class TestDecayFunctions(unittest.TestCase):
    def test_decay_functions_consistency(self):
        # Test parameters
        steps = [0, 5, 10, 50, 100]
        max_ema_decay = 0.9999
        min_ema_decay = 0.0
        ema_inv_gamma = 1.0
        ema_decay_power = 2 / 3
        use_ema_warmup = False
        start_ema_update_after_n_steps = 10.0

        for step in steps:
            decay1 = get_decay1(
                step,
                max_ema_decay,
                min_ema_decay,
                ema_inv_gamma,
                ema_decay_power,
                use_ema_warmup,
                start_ema_update_after_n_steps
            )
            decay2 = get_decay2(
                step,
                max_ema_decay,
                min_ema_decay,
                ema_inv_gamma,
                ema_decay_power,
                use_ema_warmup,
                start_ema_update_after_n_steps
            )
            self.assertAlmostEqual(decay1, decay2, places=5, msg=f"Decay mismatch at step {step}")


class TestDecayAgainstReference(unittest.TestCase):
    def test_decay_against_reference(self):
        # Test parameters
        steps = [0, 5, 10, 50, 100, 500, 1000]
        max_ema_decay = 0.9999
        min_ema_decay = 0.0
        ema_inv_gamma = 1.0
        ema_decay_power = 2 / 3
        use_ema_warmup = True
        start_ema_update_after_n_steps = 10.0

        for step in steps:
            expected_decay = reference_decay(
                step,
                max_ema_decay,
                min_ema_decay,
                ema_inv_gamma,
                ema_decay_power,
                use_ema_warmup,
                start_ema_update_after_n_steps
            )
            actual_decay1 = get_decay1(
                step,
                max_ema_decay,
                min_ema_decay,
                ema_inv_gamma,
                ema_decay_power,
                use_ema_warmup,
                start_ema_update_after_n_steps
            )
            actual_decay2 = get_decay2(
                step,
                max_ema_decay,
                min_ema_decay,
                ema_inv_gamma,
                ema_decay_power,
                use_ema_warmup,
                start_ema_update_after_n_steps
            )
            self.assertAlmostEqual(expected_decay, actual_decay1, places=5, msg=f"Decay mismatch at step {step} for get_decay1")
            #self.assertAlmostEqual(expected_decay, actual_decay2, places=5, msg=f"Decay mismatch at step {step} for get_decay2")



if __name__ == '__main__':
      unittest.main()
