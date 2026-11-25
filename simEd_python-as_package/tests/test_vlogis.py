from simEd import set_seed, vunif
from typing import Any
from simEd.generators.generators_continuous import vlogis
from utils.printOneTest import printOneTest, Tests

################################################################################
def vlogis_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vlogis(0, location=0, scale=1))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vlogis(1, location=0, scale=1, stream=88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    try: print(vlogis(1, location=0, scale=0))
    except Exception as err: print(f"correctly caught scale error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    #
    n = 1; location = 0; scale = 1; expected = Any
    printOneTest(vlogis, n, location, scale, expected=expected, context=f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; location = 0; scale = 1; stream = 0; expected = 0.6913876997384985
    printOneTest(vlogis, n, location, scale, stream, expected=expected, context=f"with initial seed {seed}")

    set_seed(seed)
    n = 5; location = 0; scale = 1; stream = 0
    expected = [0.6913877,  -0.52996884, -0.34558355, -2.25237402,  1.42843497]
    printOneTest(vlogis, n, location, scale, stream, expected=expected, context=f"with initial seed {seed}")

    set_seed(seed)
    n = 5; location = 0; scale = 1; stream = 0; antithetic = False; as_dict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = {"u": u, "x": expected}
    printOneTest(vlogis, n, location, scale, stream, antithetic, as_dict, expected=expected, context=f"with initial seed {seed}")

    print('*' * 70)

    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; location = 0; scale = 1
    expected = 0.6913876997384985
    printOneTest(vlogis, n, location, scale, expected=expected, context=f"with initial seed {seed}, default stream 0")

    stream = 1
    n = 5; location = 2; scale = 0.5
    expected = [1.7646940303733558, 2.3613471925196667, 1.7320582438983068, 1.6116508053666072, -0.5866340865333188]
    printOneTest(vlogis, n, location, scale, stream, expected=expected, context=f"with initial seed {seed}, stream {stream}")

    stream = 1
    n = 3; location = -1; scale = 2.0
    expected = [-0.3891683680618502, -3.1399788506752158, 5.2208463674710766]
    printOneTest(vlogis, n, location, scale, stream, expected=expected, context=f"same stream {stream}")
    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; location = 0; scale = 1; stream = 0
    expected = [0.6913876997384985, -0.5299688411200502, -0.34558355157725673]
    printOneTest(vlogis, n, location, scale, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; location = 1; scale = 1; stream = 1
    expected = [0.5293880607467116, 1.722694385039333, 0.4641164877966134]
    printOneTest(vlogis, n, location, scale, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; location = 0; scale = 2; stream = 2
    expected = [0.39339424585926464, 3.4504256985420905, 4.050636505156782]
    printOneTest(vlogis, n, location, scale, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    print('*' * 70)
    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[0.6913876997384985, -0.5299688411200502, -0.34558355157725673],
                [0.5293880607467116, 1.722694385039333, 0.4641164877966134],
                [0.39339424585926464, 3.4504256985420905, 4.050636505156782]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        location = 0 if stream == 0 else 1 if stream == 1 else 0
        scale = 1 if stream == 0 else 1 if stream == 1 else 2
        printOneTest(vlogis, 1, location, scale, stream, expected=expected[stream][i // 3])
    ##########################################
    ##########################################
    # now let's test antithetic variates

    print('*' * 70)
    seed = 8675309
    location = 1; scale = 0.8; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [1.5531101597907988, 0.5760249271039599, 0.7235331587381946]
    printOneTest(vlogis, 3, location, scale, stream, antithetic, expected=expected1, context=f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [0.4468898402092012, 1.42397507289604, 1.2764668412618054]
    printOneTest(vlogis, 3, location, scale, stream, antithetic, expected=expected2, context=f"seed {seed}, antithetic {antithetic}")
if __name__ == "__main__":
    vlogis_tests()