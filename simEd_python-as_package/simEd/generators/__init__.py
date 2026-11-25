# generators/__init__.py
from simEd.generators.rng_mt19937 import rng
################################################################################
# SET THE INITIAL STATE OF THE RNG 
#   if no seed: fresh, unpredictable entropy will be pulled from the OS
#   https://numpy.org/doc/2.2/reference/random/bit_generators/mt19937.html
rng.set_seed(None)  # default rng state will be "fresh, unpredictable entropy"
from .generators import set_seed, _generateUniforms
from .generators_continuous import vexp, vunif, vnorm
from .generators_discrete import vgeom, vbinom, vpois, vnbinom
from ..utils._error_checks import _checkType, _checkRange
