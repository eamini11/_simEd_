# use scipy for R-style quantile inversion via ppf function
# https://docs.scipy.org/doc/scipy/reference/stats.html#discrete-distributions
from scipy.stats import binom, \
                        geom, \
                        nbinom, \
                        poisson

from ..generators import rng, _generateUniforms
from ..utils._error_checks import _checkType, _checkRange

################################################################################
def vbinom(n:          int, 
           size:       int,
           prob:       float,
           stream:     int   = 0,
           antithetic: bool  = False,
           as_dict:    bool  = False,
        ) -> int | list[int] | dict[list[float], list[int]]:
    ''' variate generator for the binomial distribution
    Parameters:
        n:          number of observations
        size:       number of trials (zero or more)
        prob:       probability of success on each trial (0 < ‘prob’ <= 1
        stream:     an integer in [0,127] indicating the rng stream to use
        antithetic: if False (default), inverts u = U(0,1) variate(s);
            otherwise, uses 1 - u
        as_dict:    if False (default) return only the generated random
            variates; otherwise, return a dictionary with components suitable
            for visualizing inversion: "u" is a list of generated U(0,1)
            variates, and "x" is a list of inverted binom(size,prob) variates
    Returns:
        a single generated binomial variate, or a list of such variates, or
        a dictionary with keys "u" and "x" whose values are lists corresponding
        to the generated U(0,1) variates and the inverted binomial variates
        with corresponding size and probability
    '''
    for _ in [n, size, stream]: _checkType(_, int)
    _checkType(prob, float)
    for _ in [antithetic, as_dict]: _checkType(_, bool)

    max_stream = rng.numStreams() - 1
    _checkRange(n,      min_ = 1,                    msg = f"vbinom: must generate at least one variate")
    _checkRange(size,   min_ = 0,                    msg = f"vbinom: number of trials must be >= 0")
    _checkRange(prob,   min_ = 0, max_ = 1,          msg = f"vbinom: must have 0 < prob <= 1", include_min = False)
    _checkRange(stream, min_ = 0, max_ = max_stream, msg = f"vexp: stream must be in [0,{max_stream}]")

    # generate the scalar or list of U(0,1) variates for inverting
    u: float | list[float] = _generateUniforms(n, stream, antithetic)

    # use scipy.stat's binom.ppf (percent point function) a la R's qunif:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
    # "binom takes n and p and as shape parameters, where p is the probability
    #  of a single success and 1-p is the probability of a single failure."
    #
    # NB: if u is scalar, numpy's tolist() defaults to int
    x: int | list[int] = binom.ppf(u, n = size, p = prob).astype('int').tolist()

    # return a dictionary of u and x variates;
    if as_dict: return {"u":u, "x":x}

    # otherwise, return the x variates only
    return x

