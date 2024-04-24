# %%


import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze

# Assuming ema_update function is defined here (as provided earlier)
# from your_ema_module import ema_update  # Uncomment and modify if using module

def ema_update(
    params: FrozenDict,
    ema_params: FrozenDict,
    steps: int,
    max_ema_decay: float = 0.999,
    min_ema_decay: float = 0.5,
    ema_decay_power: float = 0.6666666,
    ema_inv_gamma: float = 1.0,
    start_ema_update_after: int = 100,
    update_ema_every: int = 10,
) -> FrozenDict:
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""

    def calculate_decay():
        decay = 1.0 - (1.0 + (steps / ema_inv_gamma)) ** (-ema_decay_power)
        return np.clip(decay, min_ema_decay, max_ema_decay)

    if steps < start_ema_update_after:
        """When EMA is not updated, return the current params"""
        return params

    if steps % update_ema_every == 0:
        decay = calculate_decay()
        decay_avg = 1.0 - decay

        return jax.tree_util.tree_map(
            lambda ema, p_new: decay_avg * ema + decay * p_new,
            ema_params,
            params,
        )

    return ema_params


def generate_noisy_sine_wave(num_points, noise_factor=0.5):
    x = np.linspace(0, 6.5, num_points)
    y = np.sin(x) + np.random.normal(scale=noise_factor, size=num_points)
    return x, y

def initialize_ema(y):
    """ Initialize the EMA parameters with the first data point """
    return tree_map(lambda x: x, y)  # Use tree_map to create a deep copy if y were more complex

def apply_ema_to_wave(y):
    num_points = len(y)
    ema_y = initialize_ema(y[:1])  # Initialize EMA with the first data point
    
    # Store EMA results
    ema_results = np.zeros(num_points)
    ema_results[0] = ema_y[0]
    
    for i in range(1, num_points):
        ema_y = ema_update(
            params=jnp.array([y[i]]),
            ema_params=ema_y,
            steps=i,
            start_ema_update_after=100,  # Start after the first step
            update_ema_every=10,  # Update EMA every step
            ema_inv_gamma=1.0,  # No gamma correction
            ema_decay_power=0.6666666,
            max_ema_decay=0.999,
            min_ema_decay=0.5  # Play with these values for different smoothing effects
        )
        ema_results[i] = ema_y[0]
    
    return ema_results


xs = jnp.linspace(0, 6.5, 200)
p_xs = jnp.array([xs] * jax.device_count() )

# Generate noisy sine wave data
noise_factor = 0.3
x, noisy_y = generate_noisy_sine_wave( 200, noise_factor=noise_factor)

# Apply EMA smoothing
smoothed_y = apply_ema_to_wave(noisy_y)


# make a deep copy of noisy_y
def indentity(x):
  return x

p = jax.tree_util.tree_map(indentity, noisy_y)
ema = jax.tree_util.tree_map(indentity, noisy_y)

ema = ema_update(p ,ema, 1) 
for i in range(2, 200):
  new_ema = ema_update(p ,ema, i) 
  ema = new_ema
  


# %%

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(x, noisy_y, label='Noisy Sine Wave', linestyle='-', color='blue', alpha=0.9)
if smoothed_y is not None:
    plt.plot(x, smoothed_y, label='Smoothed with EMA', linestyle='-', color='red', alpha=0.8)

plt.plot(x, ema, label='Smoothed with EMA', linestyle='--', color='purple', alpha=0.8)
plt.legend()
plt.title('Smoothing a Noisy Sine Wave using EMA')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
