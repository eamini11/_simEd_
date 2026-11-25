'''
Tests for the vunif function in the simEd package.

SETUP REQUIRED:
Before running this test, install the simEd package in development mode:
1. Navigate to the top-level simEd/ directory (where setup.py is located)
2. Run: python -m pip install -e .
3. The package will be installed as an editable/development installation

This allows importing 'from simEd import ...' to work from any directory.

To run this set of tests:
    python test_vunif.py
'''

from simEd import vunif, set_seed
from typing import Any

from utils.printOneTest import printOneTest, Tests

################################################################################
def vunif_tests() -> None:

    #############################################################
    # test exceptions
    #
    try: print(vunif(0, 0, 1))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vunif(1, 0, 1, stream = 88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    print('*' * 70)

    #############################################################
    # test exceptions
    #
    seed = 8675309
    set_seed(seed)
    n = 1; a = 0; b = 1; stream =  0; expected = 0.6662755563376126
    printOneTest(vunif, n, a, b, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; a = 0; b = 1; stream = 0
    expected = [0.66627556,0.37052416,0.41445381,0.09514488,0.80665735]
    printOneTest(vunif, n, a, b, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; a = 0; b = 1; stream = 0; antithetic = False; as_dict = True
    expected = {"u":expected, "x":expected}
    printOneTest(vunif, n, a, b, stream, antithetic, as_dict, expected = expected, context = f"with initial seed {seed}")

    print('*' * 70)

    # test no initialize streams
    n = 1; a = 10; b = 20; expected = Any
    printOneTest(vunif, n, a, b, expected = expected, context = f"with no initial seed")

    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; a = 10; b = 20 
    expected = 16.662755563376127  # use expected_tests.py
    printOneTest(vunif, n, a, b, expected = expected, context = f"with initial seed {seed}, default stream 0")

    stream = 1
    n = 5; a = 13.2; b = 19.7
    expected = [15.6990642,17.57580041,15.59944705,15.24770731,13.23662164]
    printOneTest(vunif, n, a, b, stream, expected = expected, context = f"with initial seed {seed}, stream {stream}")

    stream = 1
    n = 3; a = -3.1; b = 4.22
    expected = [1.11460654, -1.2304347, 3.90758715]
    printOneTest(vunif, n, a, b, stream, expected = expected, context = f"same stream {stream}")

    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; a = 0; b = 10; stream = 0
    expected = [6.66275556, 3.70524155, 4.14453806]
    printOneTest(vunif, n, a, b, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; a = -10; b = 0; stream = 1
    expected = [-6.15528584, -3.26799937, -6.30854299]
    printOneTest(vunif, n, a, b, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; a = -3; b = 3; stream = 2
    expected = [0.29409808, 2.0927944, 2.30057885]
    printOneTest(vunif, n, a, b, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    print('*' * 70)

    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[6.66275556, 3.70524155, 4.14453806],
                [-6.15528584, -3.26799937, -6.30854299],
                [0.29409808, 2.0927944, 2.30057885]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        a,b = (0,10) if stream == 0 else (-10,0) if stream == 1 else (-3,3)
        printOneTest(vunif, 1, a, b, stream, expected = expected[stream][i // 3])

    ##########################################
    ##########################################
    # now let's test antithetic variates

    print('*' * 70)
    seed = 8675309
    a = -2.34; b = 8.61

    set_seed(seed)
    antithetic = False
    expected1 = [4.95571734, 1.7172395, 2.19826918]
    printOneTest(vunif, 3, a, b, 0, antithetic, expected = expected1, context = f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [1.31428266, 4.5527605, 4.07173082]
    printOneTest(vunif, 3, a, b, 0, antithetic, expected = expected2, context = f"seed {seed}, antithetic {antithetic}")

    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u":u1, "x":expected1}
    printOneTest(vunif, 3, a, b, 0, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u":u2, "x":expected2}
    printOneTest(vunif, 3, a, b, 0, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vunif_tests()
