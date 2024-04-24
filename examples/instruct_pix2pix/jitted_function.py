
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
