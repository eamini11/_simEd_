from simEd import set_seed, vunif
from simEd.generators.generators_discrete import vgeom
from typing import Any
from utils.printOneTest import printOneTest, Tests

################################################################################
def vgeom_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vgeom(0, prob = 0.5))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vgeom(1, prob = 0))
    except Exception as err: print(f"correctly caught prob error: {err}")

    try: print(vgeom(1, prob = 1.5))
    except Exception as err: print(f"correctly caught prob error: {err}")

    try: print(vgeom(1, prob = 0.5, stream = 88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    #
    n = 1; prob = 0.5; expected = Any
    printOneTest(vgeom, n, prob, expected = expected, context = f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; prob = 0.5; stream = 0; expected = 2
    printOneTest(vgeom, n, prob, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; prob = 0.5; stream = 0
    expected = [2, 1, 1, 1, 3]
    printOneTest(vgeom, n, prob, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; prob = 0.5; stream = 0; antithetic = False; as_dict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = {"u":u, "x":expected}
    printOneTest(vgeom, n, prob, stream, antithetic, as_dict, expected = expected, context = f"with initial seed {seed}")

    print('*' * 70)

    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; prob = 0.3
    expected = 4
    printOneTest(vgeom, n, prob, expected = expected, context = f"with initial seed {seed}, default stream 0")

    stream = 1
    n = 5; prob = 0.2
    expected = [3, 6, 3, 2, 1]
    printOneTest(vgeom, n, prob, stream, expected = expected, context = f"with initial seed {seed}, stream {stream}")

    stream = 1
    n = 3; prob = 0.7
    expected = [1, 1, 3]
    printOneTest(vgeom, n, prob, stream, expected = expected, context = f"same stream {stream}")

    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; prob = 0.5; stream = 0
    expected = [2, 1, 1]
    printOneTest(vgeom, n, prob, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; prob = 0.5; stream = 1
    expected = [1, 2, 1]
    printOneTest(vgeom, n, prob, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; prob = 0.5; stream = 2
    expected = [2, 3, 4]
    printOneTest(vgeom, n, prob, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    print('*' * 70)
    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[2, 1, 1],
                [1, 2, 1],
                [2, 3, 4]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        prob = 0.5 if stream == 0 else 0.5 if stream == 1 else 0.5
        printOneTest(vgeom, 1, prob, stream, expected = expected[stream][i // 3])

    ##########################################
    ##########################################
    # now let's test antithetic variates
    print('*' * 70)
    seed = 8675309
    prob = 0.4; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [3, 1, 2]
    printOneTest(vgeom, 3, prob, stream, antithetic, expected = expected1, context = f"seed {seed}, antithetic {antithetic}")
   

    set_seed(seed)
    antithetic = True
    expected2 = [1, 2, 2]
    printOneTest(vgeom, 3, prob, stream, antithetic, expected = expected2, context = f"seed {seed}, antithetic {antithetic}")

    # note: with seed 8675309 and stream 0, u1 and u2 should be the same in vunif, vgeom, ...
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u":u1, "x":expected1}
    printOneTest(vgeom, 3, prob, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u":u2, "x":expected2}
    printOneTest(vgeom, 3, prob, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vgeom_tests()