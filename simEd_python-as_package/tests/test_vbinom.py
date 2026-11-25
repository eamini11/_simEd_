'''
Tests for the vbinom function in the simEd package.

SETUP REQUIRED:
Before running this test, install the simEd package in development mode:
1. Navigate to the top-level simEd_python/ directory (where setup.py is located)
2. Run: python -m pip install -e .
3. The package will be installed as an editable/development installation

This allows importing 'from simEd import ...' to work from any directory.

To run this set of tests:
    python test_vbinom.py
'''

from simEd import vbinom, set_seed, vunif
from typing import Any

from utils.printOneTest import printOneTest, Tests

################################################################################
def vbinom_tests() -> None:

    #############################################################
    # test exceptions
    #
    try: print(vbinom(0, size = 10, prob = 0.25))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vbinom(1, size = 10, prob = 0.25, stream = 88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    print('*' * 70)

    ######################  #######################################
    # test no initialize streams
    #
    n = 1; size = 10; prob = 0.25; expected = Any
    printOneTest(vbinom, n, size, prob, expected = expected, context = f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; size = 10; prob = 0.25; stream = 0; expected = 3
    printOneTest(vbinom, n, size, prob, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; size = 10; prob = 0.25; stream = 0
    expected = [3, 2, 2, 1, 4]
    printOneTest(vbinom, n, size, prob, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; size = 10; prob = 0.25; stream = 0; antithetic = False; as_dict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = {"u":u, "x":expected}
    printOneTest(vbinom, n, size, prob, stream, antithetic, as_dict, expected = expected, context = f"with initial seed {seed}")

    print('*' * 70)

    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; size = 5; prob = 0.5
    expected = 3
    printOneTest(vbinom, n, size, prob, expected = expected, context = f"with initial seed {seed}, default stream 0")

    stream = 1
    n = 5; size = 7; prob = 0.35
    expected = [2, 3, 2, 2, 0]
    printOneTest(vbinom, n, size, prob, stream, expected = expected, context = f"with initial seed {seed}, stream {stream}")

    stream = 1
    n = 3; size = 10; prob = 0.25
    expected = [3, 2, 5]
    printOneTest(vbinom, n, size, prob, stream, expected = expected, context = f"same stream {stream}")

    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; size = 5; prob = 0.5; stream = 0
    expected = [3, 2, 2]
    printOneTest(vbinom, n, size, prob, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; size = 7; prob = 0.35; stream = 1
    expected = [2, 3, 2]
    printOneTest(vbinom, n, size, prob, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; size = 10; prob = 0.25; stream = 2
    expected = [3, 4, 4]
    printOneTest(vbinom, n, size, prob, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    print('*' * 70)

    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[3, 2, 2],
                [2, 3, 2],
                [3, 4, 4]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        (size, prob) = (5,0.5) if stream == 0 else (7,0.35) if stream == 1 else (10,0.25)
        printOneTest(vbinom, 1, size, prob, stream, expected = expected[stream][i // 3])

    ##########################################
    ##########################################
    # now let's test antithetic variates

    print('*' * 70)
    seed = 8675309
    size = 10; prob = 0.25; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [3, 2, 2]
    printOneTest(vbinom, 3, size, prob, stream, antithetic, expected = expected1, context = f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [2, 3, 3]
    printOneTest(vbinom, 3, size, prob, stream, antithetic, expected = expected2, context = f"seed {seed}, antithetic {antithetic}")

    # note: with seed 8675309 and stream 0, u1 and u2 should be the same in vunif, vbinom, ...
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u":u1, "x":expected1}
    printOneTest(vbinom, 3, size, prob, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u":u2, "x":expected2}
    printOneTest(vbinom, 3, size, prob, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vbinom_tests()
