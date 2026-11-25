from simEd import set_seed, vunif
from typing import Any
from simEd.generators.generators_continuous import vgamma
from utils.printOneTest import printOneTest, Tests
################################################################################
def vgamma_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vgamma(0, shape=1, rate=1))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vgamma(1, shape=1, rate=1, stream=88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    try: print(vgamma(1, shape=0, rate=1))
    except Exception as err: print(f"correctly caught shape error: {err}")

    try: print(vgamma(1, shape=1, rate=0))
    except Exception as err: print(f"correctly caught rate error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    n = 1; shape = 2; rate = 1; expected = Any
    printOneTest(vgamma, n, shape, rate, expected=expected, context=f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; shape = 2; rate = 1; stream = 0; expected = 2.2875963507975885
    printOneTest(vgamma, n, shape, rate, stream, expected=expected, context=f"with initial seed {seed}")

    set_seed(seed)
    n = 5; shape = 2.0; rate = 0.5; stream = 0
    expected = [1.1437981753987942, 0.6462595102556908, 0.709128701073813, 0.2580818541885111, 1.519691651551262]
    printOneTest(vgamma, n, shape, rate, stream, expected=expected, context=f"with initial seed {seed}")

    set_seed(seed)
    n = 5; shape = 2; rate = 0.5; stream = 0; antithetic = False; as_dict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = {"u": u, "x": expected}
    printOneTest(vgamma, n, shape, rate, stream, antithetic, as_dict, expected=expected, context=f"with initial seed {seed}")
    print('*' * 70)

    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; shape = 2; rate = 1
    expected = 2.2875963507975885
    printOneTest(vgamma, n, shape, rate, expected=expected, context=f"with initial seed {seed}, default stream 0")

    stream = 1
    n = 5; shape = 1.5; rate = 1.0
    expected = [0.8986364020320418, 1.7269143609193494, 0.863711600459567, 0.7441612428004882, 0.038878781829415454]
    printOneTest(vgamma, n, shape, rate, stream, expected=expected, context=f"with initial seed {seed}, stream {stream}")
    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; shape = 2.0; rate = 1.0; stream = 0
    expected = [2.2875963507975885, 1.2925190205113817, 1.418257402147626]
    printOneTest(vgamma, n, shape, rate, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; shape = 1.0; rate = 2.0; stream = 1
    expected = [0.970547788367946, 2.2368142166705614, 0.9213606952926522]
    printOneTest(vgamma, n, shape, rate, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; shape = 0.5; rate = 0.5; stream = 2
    expected = [0.1420446173435845, 0.5150232715462711, 0.6157021084973832]
    printOneTest(vgamma, n, shape, rate, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    print('*' * 70)
    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[2.2875963507975885, 1.2925190205113817, 1.418257402147626],
                [0.970547788367946, 2.2368142166705614, 0.9213606952926522],
                [0.1420446173435845, 0.5150232715462711, 0.1420446173435845]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
    if stream == 0:
        shape, rate = 2.0, 1.0
    elif stream == 1:
        shape, rate = 1.0, 2.0
    else:
        shape, rate = 0.5, 0.5
    printOneTest(vgamma, 1, shape, rate, stream, expected=expected[stream][i // 3])
    ##########################################
    ##########################################
    # now let's test antithetic variates
    print('*' * 70)
    seed = 8675309
    shape = 1.5; rate = 0.8; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [0.46781871838856565, 3.1819776684808163, 1.501796314022605]
    printOneTest(vgamma, 3, shape, rate, stream, antithetic, expected=expected1, context=f"seed {seed}, antithetic {antithetic}")
    input()

    set_seed(seed)
    antithetic = True
    expected2 = [0.981073273316574, 1.9625653721107106, 1.7846531938253076]
    printOneTest(vgamma, 3, shape, rate, stream, antithetic, expected=expected2, context=f"seed {seed}, antithetic {antithetic}")

    # note: with seed 8675309 and stream 0, u1 and u2 should be the same in vunif, vgamma, ...
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u": u1, "x": expected1}
    printOneTest(vgamma, 3, shape, rate, stream, antithetic, as_dict, expected=expected, context=f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u": u2, "x": expected2}
    printOneTest(vgamma, 3, shape, rate, stream, antithetic, as_dict, expected=expected, context=f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vgamma_tests()