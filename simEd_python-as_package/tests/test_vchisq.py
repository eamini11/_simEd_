from simEd import set_seed, vunif
from typing import Any
from simEd.generators.generators_continuous import vchisq
from utils.printOneTest import printOneTest, Tests

################################################################################
def vchisq_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vchisq(0, df=1))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vchisq(1, df=1, stream=88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    try: print(vchisq(1, df=0))
    except Exception as err: print(f"correctly caught df error: {err}")

    try: print(vchisq(1, df=-1))
    except Exception as err: print(f"correctly caught negative df error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    #
    n = 1; df = 5; expected = Any
    printOneTest(vchisq, n, df, expected=expected, context=f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; df = 5.0; stream = 0; expected = 7.027344890046393
    print