
from simEd import set_seed, vunif
from typing import Any
from simEd.generators.generators_discrete import vpois

from utils.printOneTest import printOneTest, Tests

################################################################################
def vpois_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vpois(0, mu = 2.0))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vpois(1, mu = 0))
    except Exception as err: print(f"correctly caught mu error: {err}")

    try: print(vpois(1, mu = -1))
    except Exception as err: print(f"correctly caught mu error: {err}")

    try: print(vpois(1, mu = 2.0, stream = 88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    #
    n = 1; mu = 2.0; expected = Any
    printOneTest(vpois, n, mu, expected = expected, context = f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; mu = 2.0; stream = 0; expected = 2
    printOneTest(vpois, n, mu, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; mu = 2.0; stream = 0
    expected = [2, 1, 2, 0, 3]
    printOneTest(vpois, n, mu, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; mu = 2.0; stream = 0; antithetic = False; as_dict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = {"u":u, "x":expected}
    printOneTest(vpois, n, mu, stream, antithetic, as_dict, expected = expected, context = f"with initial seed {seed}")
    print('*' * 70)

    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; mu = 1.5
    expected = 2
    printOneTest(vpois, n, mu, expected = expected, context = f"with initial seed {seed}, default stream 0")

    stream = 1
    n = 5; mu = 3.0
    expected = [2, 4, 2, 2, 0]
    printOneTest(vpois, n, mu, stream, expected = expected, context = f"with initial seed {seed}, stream {stream}")

    stream = 1
    n = 3; mu = 0.5
    expected = [0, 0, 2]
    printOneTest(vpois, n, mu, stream, expected = expected, context = f"same stream {stream}")
    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; mu = 2.0; stream = 0
    expected = [2, 1, 2]
    printOneTest(vpois, n, mu, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; mu = 2.0; stream = 1
    expected = [1, 2, 1]
    printOneTest(vpois, n, mu, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; mu = 2.0; stream = 2
    expected = [2, 3, 4]
    printOneTest(vpois, n, mu, stream, expected = expected, context = f"seed {seed}, stream {stream}")
    print('*' * 70)

    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[2, 1, 2],
                [1, 2, 1],
                [2, 3, 4]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        mu = 2.0 if stream == 0 else 2.0 if stream == 1 else 2.0
        printOneTest(vpois, 1, mu, stream, expected = expected[stream][i // 3])

    ##########################################
    ##########################################
    # now let's test antithetic variates

    print('*' * 70)
    seed = 8675309
    mu = 2.5; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [3, 2, 2]
    printOneTest(vpois, 3, mu, stream, antithetic, expected = expected1, context = f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [2, 3, 3]
    printOneTest(vpois, 3, mu, stream, antithetic, expected = expected2, context = f"seed {seed}, antithetic {antithetic}")

    # note: with seed 8675309 and stream 0, u1 and u2 should be the same in vunif, vpois, ...
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u":u1, "x":expected1}
    printOneTest(vpois, 3, mu, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u":u2, "x":expected2}
    printOneTest(vpois, 3, mu, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vpois_tests()