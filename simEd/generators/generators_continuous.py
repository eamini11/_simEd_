# use scipy for R-style quantile inversion via ppf function
# https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
from scipy.stats import beta, \
                        cauchy, \
                        chisquare, \
                        expon, \
                        gamma, \
                        lognorm, \
                        logistic, \
                        norm, \
                        t, \
                        uniform, \
                        weibull_min as weibull

from .generators import rng, _generateUniforms
from ..utils._error_checks import _checkType, _checkRange
################################################################################
def vunif(n:          int, 
          min_:       float = 0.0, 
          max_:       float = 1.0,
          stream:     int   = 0,
          antithetic: bool  = False,
          as_dict:    bool  = False,
         ) -> float | list[float] | dict[list[float], list[float]]:
    ''' variate generator for the uniform distribution
    Parameters:
        n:          number of observations
        min_:       lower limit of distribution (default 0)
        max_:       upper limit of distribution (default 1)
        stream:     an integer in [0,127] indicating the rng stream to use
        antithetic: if False (default), inverts u = U(0,1) variate(s);
            otherwise, uses 1 - u
        as_dict:    if False (default) return only the generated random
            variates; otherwise, return a dictionary with components suitable
            for visualizing inversion: "u" is a list of generated U(0,1)
            variates, and "x" is a list of inverted U(min_,max_) variates
    Returns:
        a single U(min_,max_) generated variate, or a list of such variates, or
        a dictionary with keys "u" and "x" whose values are lists corresponding
        to the generated U(0,1) variates and the inverted U(min_,max_) variates
    '''
    for _ in [n, stream]:  _checkType(_, int)
    for _ in [min_, max_]: _checkType(_, float)
    for _ in [antithetic, as_dict]: _checkType(_, bool)

    max_stream = rng.numStreams() - 1
    _checkRange(n,      min_ = 1,                       msg = f"vunif: must generate at least one variate")
    _checkRange(max_,   min_ = min_,                    msg = f"vunif: must have min_ <= max_")
    _checkRange(stream, min_ = 0,    max_ = max_stream, msg = f"vunif: stream must be in [0,{max_stream}]")

    # generate the scalar or list of U(0,1) variates for inverting
    u: float | list[float] = _generateUniforms(n, stream, antithetic)

    # use scipy.stat's uniform.ppf (percent point function) a la R's qunif to
    # invert u:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
    # "Using the parameters loc and scale, one obtains the uniform distribution
    #  on [loc, loc + scale]." <-- a bit odd for uniform, but oh well
    #
    # NB: if u is scalar, numpy's tolist() defaults to float 
    x: float | list[float] = uniform.ppf(u, loc = min_, scale = max_ - min_).tolist()

    # return a dictionary of u and x variates;
    if as_dict: return {"u":u, "x":x}

    # otherwise, return the x variates only
    return x


################################################################################
def vexp(n:          int, 
         rate:       float = 1.0, 
         stream:     int   = 0,
         antithetic: bool  = False,
         as_dict:    bool  = False,
        ) -> float | list[float] | dict[list[float], list[float]]:
    ''' variate generator for the exponential distribution
    Parameters:
        n:          number of observations
        rate:       rate of distribution (default 1)
        stream:     an integer in [0,127] indicating the rng stream to use
        antithetic: if False (default), inverts u = U(0,1) variate(s);
            otherwise, uses 1 - u
        as_dict:    if False (default) return only the generated random
            variates; otherwise, return a dictionary with components suitable
            for visualizing inversion: "u" is a list of generated U(0,1)
            variates, and "x" is a list of inverted exp(rate) variates
    Returns:
        a single generated exponential variate, or a list of such variates, or
        a dictionary with keys "u" and "x" whose values are lists corresponding
        to the generated U(0,1) variates and the inverted exponential variates
        with corresponding rate
    '''
    for _ in [n, stream]:  _checkType(_, int)
    _checkType(rate, float | int)
    for _ in [antithetic, as_dict]: _checkType(_, bool)

    max_stream = rng.numStreams() - 1
    _checkRange(n,      min_ = 1, msg = f"vexp: must generate at least one variate")
    _checkRange(rate,   min_ = 0, msg = f"vexp: must have rate > 0", include_min = False)
    _checkRange(stream, min_ = 0, max_ = max_stream, msg = f"vexp: stream must be in [0,{max_stream}]")

    # generate the scalar or list of U(0,1) variates for inverting
    u: float | list[float] = _generateUniforms(n, stream, antithetic)

    # use scipy.stat's expon.ppf (percent point function) a la R's qunif:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
    # "A common parameterization for expon is in terms of the rate parameter
    #  lambda, such that pdf = lambda * exp(-lambda * x). This parameterization
    #  corresponds to using scale = 1 / lambda.
    # "Using the parameters loc and scale, one obtains the uniform distribution
    #  on [loc, loc + scale]." <-- a bit odd for uniform, but oh well
    #
    # NB: if u is scalar, numpy's tolist() defaults to float 
    x: float | list[float] = expon.ppf(u, scale = 1 / rate).tolist()

    # return a dictionary of u and x variates;
    if as_dict: return {"u":u, "x":x}

    # otherwise, return the x variates only
    return x


################################################################################

def vnorm(n: int,
          mean: float = 0.0,
          sd: float = 1.0,
          stream: int = 0,
          antithetic: bool = False,
          as_dict: bool = False) -> float | list[float] | dict[list[float], list[float]]:
   
    for _ in [n, stream]: _checkType(_, int)
    _checkType(mean, float | int)
    _checkType(sd, float | int)
    for _ in [antithetic, as_dict]: _checkType(antithetic, bool)
    

    max_stream = rng.numStreams() - 1
    _checkRange(n,      min_ = 1, msg = f"vnorm: must generate at least one variate")
    _checkRange(sd,   min_ = 0, msg = f"vnorm: must have sd > 0", include_min = False)
    _checkRange(stream, min_ = 0, max_ = max_stream, msg = f"vnorm: stream must be in [0,{max_stream}]")
    
    u: float | list[float] = _generateUniforms(n, stream, antithetic)

    x: float | list[float] = norm.ppf(u, loc = mean, scale = sd).tolist()

    if as_dict:
        return {"u": u, "x": x}
    return x

vnorm(10, 2.5, 0, -1, True, True)