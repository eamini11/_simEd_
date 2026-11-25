from simEd import set_seed, vunif
from typing import Any
from simEd.generators.generators_discrete import vnbinom

from utils.printOneTest import printOneTest, Tests

################################################################################
def vnbinom_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vnbinom(0, size = 5, prob = 0.5))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vnbinom(1, size = 0, prob = 0.5))
    except Exception as err: print(f"correctly caught size error: {err}")

    try: print(vnbinom(1, size = 5, prob = 0))
    except Exception as err: print(f"correctly caught prob error: {err}")

    try: print(vnbinom(1, size = 5, prob = 1.5))
    except Exception as err: print(f"correctly caught prob error: {err}")

    try: print(vnbinom(1, size = 5, prob = 0.5, stream = 88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    n = 1; size = 5; prob = 0.5; expected = Any
    printOneTest(vnbinom, n, size, prob, expected = expected, context = f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; size = 5; prob = 0.5; stream = 0; expected = 6
    printOneTest(vnbinom, n, size, prob, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; size = 5; prob = 0.5; stream = 0
    expected = [6, 4, 4, 1, 8]
    printOneTest(vnbinom, n, size, prob, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; size = 5; prob = 0.5; stream = 0; antithetic = False; as_dict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = {"u":u, "x":expected}
    printOneTest(vnbinom, n, size, prob, stream, antithetic, as_dict, expected = expected, context = f"with initial seed {seed}")
    print('*' * 70)

    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; size = 3; prob = 0.3
    expected = 8
    printOneTest(vnbinom, n, size, prob, expected = expected, context = f"with initial seed {seed}, default stream 0")

    stream = 1
    n = 5; size = 4; prob = 0.4
    expected = [4, 7, 4, 4, 0]
    printOneTest(vnbinom, n, size, prob, stream, expected = expected, context = f"with initial seed {seed}, stream {stream}")

    stream = 1
    n = 3; size = 2; prob = 0.7
    expected = [1, 0, 3]
    printOneTest(vnbinom, n, size, prob, stream, expected = expected, context = f"same stream {stream}")
    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; size = 5; prob = 0.5; stream = 0
    expected = [6, 4, 4]
    printOneTest(vnbinom, n, size, prob, stream, expected = expected, context = f"seed {seed}, stream {stream}")
    

    n = 3; size = 5; prob = 0.5; stream = 1
    expected = [4, 6, 4]
    printOneTest(vnbinom, n, size, prob, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; size = 5; prob = 0.5; stream = 2
    expected = [5, 8, 9]
    printOneTest(vnbinom, n, size, prob, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    print('*' * 70)
    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[6, 4, 4],
                [4, 6, 4],
                [5, 8, 9]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        size = 5 if stream == 0 else 5 if stream == 1 else 5
        prob = 0.5 if stream == 0 else 0.5 if stream == 1 else 0.5
        printOneTest(vnbinom, 1, size, prob, stream, expected = expected[stream][i // 3])
    ##########################################
    ##########################################
    # now let's test antithetic variates

    print('*' * 70)
    seed = 8675309
    size = 6; prob = 0.6; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [5, 3, 3]
    printOneTest(vnbinom, 3, size, prob, stream, antithetic, expected = expected1, context = f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [3, 4, 4]
    printOneTest(vnbinom, 3, size, prob, stream, antithetic, expected = expected2, context = f"seed {seed}, antithetic {antithetic}")

    # note: with seed 8675309 and stream 0, u1 and u2 should be the same in vunif, vnbinom, ...
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]
    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u":u1, "x":expected1}
    printOneTest(vnbinom, 3, size, prob, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u":u2, "x":expected2}
    printOneTest(vnbinom, 3, size, prob, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vnbinom_tests()