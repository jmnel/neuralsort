

def exp_moving_avg(x: float,
                   ema_prev: float = None,
                   lag: int = 12,
                   decay: float = 2):

    if ema_prev is None:
        return x

    ema = x * (decay / (1. + lag))
    ema += ema_prev * (1. - decay / (1. + lag))

    return ema
