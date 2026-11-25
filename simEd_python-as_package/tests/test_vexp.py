'''
Tests for the vexp function in simEd package.

SETUP REQUIRED:
Before running this test, install the simEd package in development mode:
1. Navigate to the top-level simEd/ directory (where setup.py is located)
2. Run: python -m pip install -e .
3. The package will be installed as an editable/development installation

This allows importing 'from simEd import ...' to work from any directory.

To run this set of tests:
    python test_vexp.py
'''

from simEd import vexp, set_seed, vunif
from typing import Any

from utils.printOneTest import printOneTest, Tests

################################################################################
def vexp_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vexp(0, rate = 1))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vexp(1, rate = 1, stream = 88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    #
    n = 1; rate = 1; expected = Any
    printOneTest(vexp, n, rate, expected = expected, context = f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; rate = 0.5; stream = 0; expected = 2.1948792909915595
    printOneTest(vexp, n, rate, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; rate = 0.5; stream = 0
    expected = [2.194879, 0.9257356, 1.07042, 0.1999609, 3.286583]
    printOneTest(vexp, n, rate, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; rate = 0.5; stream = 0; antithetic = False; as_dict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = {"u":u, "x":expected}
    printOneTest(vexp, n, rate, stream, antithetic, as_dict, expected = expected, context = f"with initial seed {seed}")

    print('*' * 70)

    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; rate = 1
    expected = 1.0974396454957798
    printOneTest(vexp, n, rate, expected = expected, context = f"with initial seed {seed}, default stream 0")

    stream = 1
    n = 5; rate = 0.5
    expected = [0.9705478, 2.236814, 0.9213607, 0.756766, 0.01130006]
    printOneTest(vexp, n, rate, stream, expected = expected, context = f"with initial seed {seed}, stream {stream}")

    stream = 1
    n = 3; rate = 2.0
    expected = [0.428735, 0.1474575, 1.57702]
    printOneTest(vexp, n, rate, stream, expected = expected, context = f"same stream {stream}")

    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; rate = 0.5; stream = 0
    expected = [2.194879, 0.9257356, 1.07042]
    printOneTest(vexp, n, rate, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; rate = 1; stream = 1
    expected = [0.4852739, 1.118407, 0.4606803]
    printOneTest(vexp, n, rate, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; rate = 2; stream = 2
    expected = [0.3981621, 0.9445728, 1.074631]
    printOneTest(vexp, n, rate, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    print('*' * 70)

    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[2.194879, 0.9257356, 1.07042],
                [0.4852739, 1.118407, 0.4606803],
                [0.3981621, 0.9445728, 1.074631]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        rate = 0.5 if stream == 0 else 1 if stream == 1 else 2
        printOneTest(vexp, 1, rate, stream, expected = expected[stream][i // 3])

    ##########################################
    ##########################################
    # now let's test antithetic variates

    print('*' * 70)
    seed = 8675309
    rate = 0.675; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [1.625837, 0.6857301, 0.792904]
    printOneTest(vexp, 3, rate, stream, antithetic, expected = expected1, context = f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [0.6015584, 1.470869, 1.30488]
    printOneTest(vexp, 3, rate, stream, antithetic, expected = expected2, context = f"seed {seed}, antithetic {antithetic}")

    # note: with seed 8675309 and stream 0, u1 and u2 should be the same in vunif, vexp, ...
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u":u1, "x":expected1}
    printOneTest(vexp, 3, rate, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u":u2, "x":expected2}
    printOneTest(vexp, 3, rate, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vexp_tests()
