# use scipy for R-style quantile inversion via ppf function
# https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
import numpy as np
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
          asDict:     bool  = False,
         ) -> float | list[float] | dict[list[float], list[float]]:
    ''' variate generator for the uniform distribution
    Parameters:
        n:          number of observations
        min_:       lower limit of distribution (default 0)
        max_:       upper limit of distribution (default 1)
        stream:     an integer in [0,127] indicating the rng stream to use
        antithetic: if False (default), inverts u = U(0,1) variate(s);
            otherwise, uses 1 - u
        asDict:     if False (default) return only the generated random
            variates; otherwise, return a dictionary with components suitable
            for visualizing inversion: "u" is a list of generated U(0,1)
            variates, and "x" is a list of inverted U(min_,max_) variates
    Returns:
        a single U(min_,max_) generated variate (n = 1), or a list of such
        variates (n > 1), or a dictionary with keys "u" and "x" whose values
        are lists corresponding to the generated U(0,1) variates and the
        inverted U(min_,max_) variates
    '''
    for _ in [n, stream]:  _checkType(_, int)
    for _ in [min_, max_]: _checkType(_, float)
    for _ in [antithetic, asDict]: _checkType(_, bool)

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
    if asDict: return {"u":u, "x":x}

    # otherwise, return the x variates only
    return x

################################################################################
def vexp(n:          int, 
         rate:       float = 1.0, 
         stream:     int   = 0,
         antithetic: bool  = False,
         asDict:     bool  = False,
        ) -> float | list[float] | dict[list[float], list[float]]:
    ''' variate generator for the exponential distribution
    Parameters:
        n:          number of observations
        rate:       rate of distribution (default 1)
        stream:     an integer in [0,127] indicating the rng stream to use (default: 0)
        antithetic: if False (default), inverts u = U(0,1) variate(s);
            otherwise, uses 1 - u
        asDict:     if False (default) return only the generated random
            variates; otherwise, return a dictionary with components suitable
            for visualizing inversion: "u" is a list of generated U(0,1)
            variates, and "x" is a list of inverted exp(rate) variates
    Returns:
        a single generated exponential variate (n = 1), or a list of such
        variates (n > 1), or a dictionary with keys "u" and "x" whose values
        are lists corresponding to the generated U(0,1) variates and the
        inverted exponential variates with corresponding rate
    '''
    for _ in [n, stream]:  _checkType(_, int)
    _checkType(rate, float | int)
    for _ in [antithetic, asDict]: _checkType(_, bool)

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
    if asDict: return {"u":u, "x":x}

    # otherwise, return the x variates only
    return x

################################################################################
def vnorm(n:          int,
          mean:       float = 0.0,
          sd:         float = 1.0,
          stream:     int = 0,
          antithetic: bool = False,
          asDict:     bool = False
         ) -> float | list[float] | dict[list[float], list[float]]:
    ''' variate generator for the normal (Gaussian) distribution
    
    Parameters:
        n:          number of observations
        mean:       distribution mean (default: 0.0)
        sd:         distribution standard deviation (default: 1.0, must be > 0)
        stream:     an integer in [0,127] indicating the rng stream to use (default: 0)
        antithetic: if False (default), inverts u = U(0,1) variate(s);
            otherwise, uses 1 - u
        asDict:     if False (default) return only the generated normal
            variates; otherwise, return a dictionary with components suitable
            for visualizing inversion: "u" is a list of generated U(0,1)
            variates, and "x" is a list of inverted normal(mu,sd) variates

    Returns:
        a single generated normal(mean, sd) variate (n = 1), or a list of such
        variates (n > 1), or a dictionary with keys "u" and "x" whose values
        are lists corresponding to the generated U(0,1) variates and the
        inverted normal variates with corresponding mean and sd

    '''
    for _ in [n, stream]: _checkType(_, int)
    for _ in [mean, sd]:  _checkType(_, float | int)
    for _ in [antithetic, asDict]: _checkType(antithetic, bool)

    max_stream = rng.numStreams() - 1
    _checkRange(n,      min_ = 1, msg = f"vnorm: must generate at least one variate")
    _checkRange(sd,     min_ = 0, msg = f"vnorm: must have sd > 0", include_min = False)
    _checkRange(stream, min_ = 0, max_ = max_stream, msg = f"vnorm: stream must be in [0,{max_stream}]")
    
    # generate the scalar or list of U(0,1) variates for inverting
    u: float | list[float] = _generateUniforms(n, stream, antithetic)

    # use scipy.stats's norm.ppf (percent point function) a la R's qnorm:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    # "The location (loc) keyword specifies the mean. The scale (scale) keyword 
    #  specifies the standard deviation."
    # The normal distribution uses loc for mean and scale for standard deviation,
    # making it straightforward to parameterize.
    #
    # NB: if u is scalar, numpy's tolist() defaults to float 
    x: float | list[float] = norm.ppf(u, loc = mean, scale = sd).tolist()

    # return a dictionary of u and x variates;
    if asDict: return {"u": u, "x": x}

    # otherwise, return the x variates only
    return x

################################################################################
def vgamma(n: int,
           shape: float = 1.0,
           scale: float = 1.0,
           stream: int = 0,
           antithetic: bool = False,
           asDict: bool = False) -> float | list[float] | dict[str, list[float]]:
    '''variate generator for the gamma distribution

    Parameters:
        n:          number of observations
        shape:      shape parameter (default 1.0, must be > 0)
        scale:      scale parameter (default 1.0, must be > 0)
        stream:     an integer in [0,127] indicating the rng stream to use
        antithetic: if False (default), uses U(0,1) variates directly;
                    if True, inverts as 1 - u
        asDict:     if False (default), return only gamma variates;
                    if True, return a dictionary with "u" and "x" values

    Returns:
        A single gamma variate (if n = 1), a list of gamma variates (if n > 1),
        or a dictionary with "u" and "x" lists suitable for inversion visualization.
    '''
    for _ in [n, stream]: _checkType(_, int)
    for _ in [shape, scale]: _checkType(_, float | int)
    for _ in [antithetic, asDict]: _checkType(_, bool)

    max_stream = rng.numStreams() - 1
    _checkRange(n,      min_=1, msg="vgamma: must generate at least one variate")
    _checkRange(shape,  min_=0, msg="vgamma: must have shape > 0", include_min=False)
    _checkRange(scale,  min_=0, msg="vgamma: must have scale > 0", include_min=False)
    _checkRange(stream, min_=0, max_=max_stream, msg=f"vgamma: stream must be in [0, {max_stream}]")

    # generate U(0,1) variates using specified stream and antithetic flag
    u: float | list[float] = _generateUniforms(n, stream, antithetic)

    # use scipy.stats.gamma.ppf for inversion
    # gamma.ppf(u, a=shape, scale=scale) gives the quantile x such that P(X ≤ x) = u
    x: float | list[float] = gamma.ppf(u, a=shape, scale=scale).tolist()

    # return inversion dictionary or variates only
    if asDict:
        return {"u": u, "x": x}
    return x

################################################################################
def vt(n: int,
       df: float,
       stream: int = 0,
       antithetic: bool = False,
       asDict: bool = False) -> float | list[float] | dict[str, list[float]]:
    '''
    Generate n values from the Student's t-distribution with df degrees of freedom.
    Parameters:
    n: number of random values to generate
    df: degrees of freedom
    stream: RNG stream index
    antithetic: whether to use antithetic variates
    as_dict: whether to return {"u": [...], "x": [...]} instead of just x
    Returns:
    A float (if n==1), list[float], or a dictionary with uniform samples and t-values
    '''
    _checkType(n, int)
    _checkType(df, float | int)
    _checkRange(df, min_=0, msg="vt: degrees of freedom must be > 0", include_min=False)
    
    u = vunif(n, 0, 1, stream=stream, antithetic=antithetic)
    x = t.ppf(u, df)
    
    if n == 1:
        # Handle scalar case - x is already a scalar when n=1
        if hasattr(x, '__len__') and len(x) == 1:
            x = float(x[0])
        else:
            x = float(x)
        
        if hasattr(u, '__len__') and len(u) == 1:
            u = float(u[0])
        else:
            u = float(u)
            
        if asDict:
            return {"u": u, "x": x}
        return x
    else:
        # For n > 1, convert to list
        if asDict:
            return {"u": list(u), "x": list(x)}
        return list(x)

################################################################################
def vlogis(n: int,
           location: float = 0.0,
           scale: float = 1.0,
           stream: int = 0,
           antithetic: bool = False,
           asDict: bool = False) -> float | list[float] | dict[str, list[float]]:
    '''variate generator for the logistic distribution

    Parameters:
        n:          number of observations (must be >= 1)
        location:   location parameter (default 0.0)
        scale:      scale parameter (default 1.0, must be > 0)
        stream:     an integer in [0,127] indicating the rng stream to use
        antithetic: if False (default), uses U(0,1) variates directly;
                    if True, inverts as 1 - u
        asDict:     if False (default), return only logistic variates;
                    if True, return a dictionary with "u" and "x" values

    Returns:
        A single logistic variate (if n = 1), a list of logistic variates (if n > 1),
        or a dictionary with "u" and "x" lists suitable for inversion visualization.
    '''
    for _ in [n, stream]: _checkType(_, int)
    for _ in [location, scale]: _checkType(_, float | int)
    for _ in [antithetic, asDict]: _checkType(_, bool)

    max_stream = rng.numStreams() - 1
    _checkRange(n,       min_=1, msg="vlogis: must generate at least one variate")
    _checkRange(scale,   min_=0, msg="vlogis: scale must be > 0", include_min=False)
    _checkRange(stream,  min_=0, max_=max_stream, msg=f"vlogis: stream must be in [0, {max_stream}]")

    # generate U(0,1) variates using specified stream and antithetic flag
    u: float | list[float] = _generateUniforms(n, stream, antithetic)

    # transform U(0,1) to logistic distribution using inverse CDF (logit function)
    x: float | list[float] = logistic.ppf(u, loc=location, scale=scale)
    if isinstance(x, np.ndarray):
        x = x.tolist()
    # return inversion dictionary or variates only
    if asDict:
        return {"u": u, "x": x}
    return x


################################################################################

def vcauchy(n: int,
            location: float = 0.0,
            scale: float = 1.0,
            stream: int = 0,
            antithetic: bool = False,
            asDict: bool = False) -> float | list[float] | dict[str, list[float]]:
    '''variate generator for the Cauchy distribution using inverse transform'''
    _checkType(n, int)
    _checkType(location, float | int)
    _checkType(scale, float | int)
    _checkType(stream, int)
    _checkType(antithetic, bool)
    _checkType(asDict, bool)

    _checkRange(n, min_=1, msg="vcauchy: must generate at least one variate")
    _checkRange(scale, min_=0, msg="vcauchy: scale must be > 0", include_min=False)
    _checkRange(stream, min_=0, max_=rng.numStreams() - 1, msg="vcauchy: invalid stream")

    u = _generateUniforms(n, stream, antithetic)
    x = cauchy.ppf(u, loc=location, scale=scale)
    if isinstance(x, np.ndarray): x = x.tolist()
    return {"u": u, "x": x} if asDict else x

################################################################################

def vweibull(n: int,
             shape: float = 1.0,
             scale: float = 1.0,
             stream: int = 0,
             antithetic: bool = False,
             asDict: bool = False) -> float | list[float] | dict[str, list[float]]:
    '''variate generator for the Weibull distribution using inverse transform'''
    _checkType(n, int)
    _checkType(shape, float | int)
    _checkType(scale, float | int)
    _checkType(stream, int)
    _checkType(antithetic, bool)
    _checkType(asDict, bool)

    _checkRange(n, min_=1, msg="vweibull: must generate at least one variate")
    _checkRange(shape, min_=0, msg="vweibull: shape must be > 0", include_min=False)
    _checkRange(scale, min_=0, msg="vweibull: scale must be > 0", include_min=False)
    _checkRange(stream, min_=0, max_=rng.numStreams() - 1, msg="vweibull: invalid stream")

    u = _generateUniforms(n, stream, antithetic)
    u_array = np.asarray(u)

    result = np.full_like(u_array, 0.0, dtype=float)
    result[u_array == 0] = 0.0
    result[u_array == 1] = np.inf

    mask_valid = (u_array > 0) & (u_array < 1)
    result[mask_valid] = scale * np.power(-np.log(1 - u_array[mask_valid]), 1.0 / shape)

    x = result.tolist()
    return {"u": u, "x": x} if asDict else x

################################################################################

def Vchisq(n: int,
           df: float,
           stream: int = 0,
           antithetic: bool = False,
           asDict: bool = False) -> float | list[float] | dict[str, list[float]]:
    '''variate generator for the Chi-squared distribution using inverse transform'''
    _checkType(n, int)
    _checkType(df, float | int)
    _checkType(stream, int)
    _checkType(antithetic, bool)
    _checkType(asDict, bool)

    _checkRange(n, min_=1, msg="Vchisq: must generate at least one variate")
    _checkRange(df, min_=0, msg="Vchisq: degrees of freedom must be > 0", include_min=False)
    _checkRange(stream, min_=0, max_=rng.numStreams() - 1, msg="Vchisq: invalid stream")

    u = _generateUniforms(n, stream, antithetic)
    x = chisquare.ppf(u, df)
    if isinstance(x, np.ndarray): x = x.tolist()
    return {"u": u, "x": x} if asDict else x



################################################################################

def Vlnorm(p: float | list[float],
           meanlog: float = 0.0,
           sdlog: float = 1.0) -> float | list[float]:
    '''inverse log-normal distribution function (quantile function)
    
    Parameters:
    p: probability value(s) in [0,1] for which to compute quantiles
    meanlog: mean of the underlying normal distribution (default 0.0)
    sdlog: standard deviation of the underlying normal distribution (default 1.0, must be > 0)
    
    Returns:
    quantile value(s) corresponding to probability p for LogNormal(meanlog, sdlog)
    
    Mathematical Formula: Q(p) = exp(meanlog + sdlog * Φ^(-1)(p))
    '''
    # STEP 1: Type checking
    _checkType(p, float | int | list[float] | list[int])
    _checkType(meanlog, float | int)
    _checkType(sdlog, float | int)
    
    # STEP 2: Convert and validate inputs
    p_array = np.asarray(p)
    if np.any((p_array < 0) | (p_array > 1)):  # Fixed: p_array not np.array
        raise ValueError("ilnorm: probability values must be in [0,1]")
    _checkRange(sdlog, min_=0, msg="ilnorm: sdlog parameter must be > 0", include_min=False)
    
    # STEP 3: Apply log-normal quantile formula
    # METHOD A: Using mathematical relationship (recommended)
    normal_quantiles = lognorm.ppf(p_array, loc=meanlog, scale=sdlog)
    result = np.exp(normal_quantiles)
    
    # METHOD B: Using scipy directly (alternative)
    # result = lognorm.ppf(p_array, s=sdlog, scale=np.exp(meanlog))
    
    # STEP 4: Handle return format
    if np.isscalar(p):
        return float(result)
    return result.tolist()


################################################################################

def vbeta(n: int,
          alpha: float = 1.0,
          beta_param: float = 1.0,
          stream: int = 0,
          antithetic: bool = False,
          asDict: bool = False) -> float | list[float] | dict[str, list[float]]:
    '''variate generator for the beta distribution

    Parameters:
        n:          number of observations
        alpha:      first shape parameter (default 1.0, must be > 0)
        beta_param: second shape parameter (default 1.0, must be > 0)
        stream:     an integer in [0,127] indicating the rng stream to use
        antithetic: if False (default), uses U(0,1) variates directly;
                    if True, inverts as 1 - u
        asDict:     if False (default), return only beta variates;
                    if True, return a dictionary with "u" and "x" values

    Returns:
        A single beta variate (if n = 1), a list of beta variates (if n > 1),
        or a dictionary with "u" and "x" lists suitable for inversion visualization.
    '''
    for _ in [n, stream]: _checkType(_, int)
    for _ in [alpha, beta_param]: _checkType(_, float | int)
    for _ in [antithetic, asDict]: _checkType(_, bool)

    max_stream = rng.numStreams() - 1
    _checkRange(n,          min_=1, msg="vbeta: must generate at least one variate")
    _checkRange(alpha,      min_=0, msg="vbeta: must have alpha > 0", include_min=False)
    _checkRange(beta_param, min_=0, msg="vbeta: must have beta > 0", include_min=False)
    _checkRange(stream,     min_=0, max_=max_stream, msg=f"vbeta: stream must be in [0, {max_stream}]")

    # generate U(0,1) variates using specified stream and antithetic flag
    u: float | list[float] = _generateUniforms(n, stream, antithetic)

    # use scipy.stats.beta.ppf for inversion
    # beta.ppf(u, a=alpha, b=beta_param) gives the quantile x such that P(X ≤ x) = u
    x: float | list[float] = beta.ppf(u, a=alpha, b=beta_param).tolist()

    # return inversion dictionary or variates only
    if asDict:
        return {"u": u, "x": x}
    return x
