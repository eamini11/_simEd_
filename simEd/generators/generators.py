from rng_mt19937 import rng
from ..utils._error_checks import _checkType
################################################################################
def set_seed(seed: int | None = None) -> None:
    ''' function to set the state of the MT19937 generator and its streams
    Parameters:
        seed: an integer initial seed, or None for system-supplied state
    '''
    if seed is not None: _checkType(seed, int)
    rng.set_seed(seed)

################################################################################
def _generateUniforms(n: int,
                      stream: int = 0, 
                      antithetic: bool = False
                     ) -> float | list[float]:
    ''' private method for generating scalar or vector of U(0,1) variates
    Parameters:
        n: number of variates to generate
        stream: integer stream in [0,127]
        antithetic: if True, uses 1-u
    Returns:
        a scalar or list of U(0,1) variates so generated
    '''
    if n == 1:
        # generate a single variate via R's qunif-style inversion
        u: float = rng.uniform(a = 0, b = 1, which_stream = stream)
        if antithetic: u = 1 - u
    else:
        # generate multiple variates via R's qunif-style inversion
        u: list[float] = []
        for _ in range(n):
            u_val = rng.uniform(a = 0, b = 1, which_stream = stream)
            if antithetic: u_val = 1 - u_val
            u.append(u_val)
    return u

