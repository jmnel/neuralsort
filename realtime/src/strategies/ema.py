from typing import Union


def exp_moving_avg(x: float,
                   prev: Union[float, None],
                   lag: int = 10,
                   decay: float = 2) -> float:

    ema = x * (decay / (1 + lag))

    if prev is not None:
        ema += prev * (1 - (decay / (1 + lag)))
    else:
        ema = x
#        ema = x

    return ema
