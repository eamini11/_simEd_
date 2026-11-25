from simEd import set_seed, vunif
from typing import Any
from simEd.generators.generators_continuous import vcauchy
from utils.printOneTest import printOneTest, Tests

################################################################################
def vcauchy_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vcauchy(0, location=0, scale=1))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vcauchy(1, location=0, scale=1, stream=88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    try: print(vcauchy(1, location=0, scale=0))
    except Exception as err: print(f"correctly caught scale error: {err}")

    try: print(vcauchy(1, location=0, scale=-1))
    except Exception as err: print(f"correctly caught negative scale error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    #
    n = 1; location = 0; scale = 1; expected = Any
    printOneTest(vcauchy, n, location, scale, expected=expected, context=f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; location = 0.0; scale = 1.0; stream = 0; expected =  0.5757131506152481
    printOneTest(vcauchy, n, location, scale, stream, expected=expected, context=f"with initial seed {seed}")

    set_seed(seed)
    n = 5; location = 0; scale = 1; stream = 0
    expected = [ 0.57571315, -0.43078499, -0.27541425, -3.24529412,  1.43872157]
    printOneTest(vcauchy, n, location, scale, stream, expected=expected, context=f"with initial seed {seed}")
    print('*' * 70)
    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; location = 0.0; scale = 1.0; stream = 0
    expected = [0.5757131506152481, -0.4307849876466338, -0.2754142481012659]
    printOneTest(vcauchy, n, location, scale, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; location = 1.0; scale = 2.0; stream = 1
    expected = [0.24046518384862925, 2.21009885505116, 0.12814239563891416]
    printOneTest(vcauchy, n, location, scale, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; location = -1.0; scale = 0.5; stream = 2
    expected = [-0.9223908893681352, -0.027780134416726554, 0.3037254574682513]
    printOneTest(vcauchy, n, location, scale, stream, expected=expected, context=f"seed {seed}, stream {stream}")
    print('*' * 70)
    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[0.5757131506152481, -0.4307849876466338, -0.2754142481012659],
                [0.24046518384862925, 2.21009885505116, 0.12814239563891416],
                [-0.9223908893681352, -0.027780134416726554, 0.3037254574682513]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        location = 0 if stream == 0 else 1 if stream == 1 else -1
        scale = 1 if stream == 0 else 2 if stream == 1 else 0.5
        printOneTest(vcauchy, 1, location, scale, stream, expected=expected[stream][i // 3])
    ##########################################
    ##########################################
    # now let's test antithetic variates
    print('*' * 70)
    seed = 8675309
    location = 0.0; scale = 1.0; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [0.5757131506152481, -0.4307849876466338, -0.2754142481012659]
    printOneTest(vcauchy, 3, location, scale, stream, antithetic, expected=expected1, context=f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [-0.5757131506152481, 0.4307849876466338, 0.2754142481012659]
    printOneTest(vcauchy, 3, location, scale, stream, antithetic, expected=expected2, context=f"seed {seed}, antithetic {antithetic}")

    # note: with seed 8675309 and stream 0, u1 and u2 should be the same in vunif, vcauchy, ...
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u": u1, "x": expected1}
    printOneTest(vcauchy, 3, location, scale, stream, antithetic, as_dict, expected=expected, context=f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u": u2, "x": expected2}
    printOneTest(vcauchy, 3, location, scale, stream, antithetic, as_dict, expected=expected, context=f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vcauchy_tests()