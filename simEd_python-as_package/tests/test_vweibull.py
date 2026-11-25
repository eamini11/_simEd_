from simEd import set_seed, vunif
from typing import Any
from simEd.generators.generators_continuous import vweibull
from utils.printOneTest import printOneTest, Tests

################################################################################
def vweibull_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vweibull(0, shape=2, scale=1))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vweibull(1, shape=2, scale=1, stream=88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    try: print(vweibull(1, shape=0, scale=1))
    except Exception as err: print(f"correctly caught shape error: {err}")

    try: print(vweibull(1, shape=2, scale=0))
    except Exception as err: print(f"correctly caught scale error: {err}")

    try: print(vweibull(1, shape=-1, scale=1))
    except Exception as err: print(f"correctly caught negative shape error: {err}")

    try: print(vweibull(1, shape=2, scale=-1))
    except Exception as err: print(f"correctly caught negative scale error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    #
    n = 1; shape = 2; scale = 1; expected = Any
    printOneTest(vweibull, n, shape, scale, expected=expected, context=f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; shape = 2; scale = 1; stream = 0; expected = 1.0475875359585851
    printOneTest(vweibull, n, shape, scale, stream, expected=expected, context=f"with initial seed {seed}")

    set_seed(seed)
    n = 5; shape = 2; scale = 1; stream = 0
    expected = [1.04758754, 0.68034388, 0.73158062, 0.31619684, 1.28190923]
    printOneTest(vweibull, n, shape, scale, stream, expected=expected, context=f"with initial seed {seed}")

    set_seed(seed)
    n = 5; shape = 2.0; scale = 1.0; stream = 0; antithetic = False; asDict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = {"u": u, "x": expected}
    printOneTest(vweibull, n, shape, scale, stream, antithetic, asDict, expected=expected, context=f"with initial seed {seed}")
    print('*' * 70)
    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; shape = 1.5; scale = 2.0
    expected = 2.127896135367395
    printOneTest(vweibull, n, shape, scale, expected=expected, context=f"with initial seed {seed}, default stream 0")
    ##########################################
    ##########################################
    # now let's test streams
    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; shape = 2.0; scale = 1.0; stream = 0
    expected = [1.0475875359585851, 0.6803438825732462, 0.7315806196992948]
    printOneTest(vweibull, n, shape, scale, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; shape = 1.5; scale = 2.0; stream = 1
    expected = [1.2350596919165437, 2.154913993929678, 1.1929707246337953]
    printOneTest(vweibull, n, shape, scale, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; shape = 3.0; scale = 0.5; stream = 2
    expected = [0.4634468909522902, 0.6180996312521562, 0.6452575628894706]
    printOneTest(vweibull, n, shape, scale, stream, expected=expected, context=f"seed {seed}, stream {stream}")
    print('*' * 70)
    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[1.0475875359585851, 0.6803438825732462, 0.7315806196992948],
                [1.2350596919165437, 2.154913993929678, 1.1929707246337953],
                [0.4634468909522902, 0.6180996312521562, 0.6452575628894706]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        shape = 2 if stream == 0 else 1.5 if stream == 1 else 3
        scale = 1 if stream == 0 else 2 if stream == 1 else 0.5
        printOneTest(vweibull, 1, shape, scale, stream, expected=expected[stream][i // 3])
    ##########################################
    ##########################################
    # now let's test antithetic variates
    print('*' * 70)
    seed = 8675309
    shape = 2.0; scale = 1.0; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [1.0475875359585851, 0.6803438825732462, 0.7315806196992948]
    printOneTest(vweibull, 3, shape, scale, stream, antithetic, expected=expected1, context=f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [0.6372220537279617, 0.9964118825440056, 0.9385061292803903]
    printOneTest(vweibull, 3, shape, scale, stream, antithetic, expected=expected2, context=f"seed {seed}, antithetic {antithetic}")

    # note: with seed 8675309 and stream 0, u1 and u2 should be the same in vunif, vweibull, ...
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u": u1, "x": expected1}
    printOneTest(vweibull, 3, shape, scale, stream, antithetic, as_dict, expected=expected, context=f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u": u2, "x": expected2}
    printOneTest(vweibull, 3, shape, scale, stream, antithetic, as_dict, expected=expected, context=f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vweibull_tests()