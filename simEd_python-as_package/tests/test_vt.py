from simEd import set_seed, vunif
from typing import Any
from simEd.generators.generators_continuous import vt
from utils.printOneTest import printOneTest, Tests

################################################################################
def vt_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vt(0, df=1))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vt(1, df=1, stream=88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    try: print(vt(1, df=0))
    except Exception as err: print(f"correctly caught df error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    #
    n = 1; df = 5; expected = Any
    printOneTest(vt, n, df, expected=expected, context=f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; df = 5.0; stream = 0; expected = 0.45609343712602307
    printOneTest(vt, n, df, stream, expected=expected, context=f"with initial seed {seed}")

    set_seed(seed)
    n = 5; df = 3; stream = 0
    expected = [0.47465006914281643, -0.36244520588409807, -0.23560416011371044, -1.6864770537189746, 1.0104950038555975]
    printOneTest(vt, n, df, stream, expected=expected, context=f"with initial seed {seed}")

    set_seed(seed)
    n = 5; df = 3.0; stream = 0; antithetic = False; as_dict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = {"u": u, "x": expected}
    printOneTest(vt, n, df, stream, antithetic, as_dict, expected=expected, context=f"with initial seed {seed}")

    print('*' * 70)
    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; df = 10.0
    expected = 0.442637219919599
    printOneTest(vt, n, df, expected=expected, context=f"with initial seed {seed}, default stream 0")
    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; df = 5.0; stream = 0
    expected = [0.45609343712602307, -0.34936454587763716, -0.2276865026226581]
    printOneTest(vt, n, df, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; df = 3.0; stream = 1
    expected = [-0.321480148459259, 0.49656805031694745, -0.3665350881512703]
    printOneTest(vt, n, df, stream, expected=expected, context=f"seed {seed}, stream {stream}")

    n = 3; df = 10.0; stream = 2
    expected = [0.12633949481386667, 1.0873340777805436, 1.2691206804923925]
    printOneTest(vt, n, df, stream, expected=expected, context=f"seed {seed}, stream {stream}")
    print('*' * 70)

    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[0.45609343712602307, -0.34936454587763716, -0.2276865026226581],
                [-0.321480148459259, 0.49656805031694745, -0.3665350881512703],
                [0.12633949481386667, 1.0873340777805436, 1.2691206804923925]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        df = 5 if stream == 0 else 3 if stream == 1 else 10
        printOneTest(vt, 1, df, stream, expected=expected[stream][i // 3])
    ##########################################
    ##########################################
    # now let's test antithetic variates

    print('*' * 70)
    seed = 8675309
    df = 4.0; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [0.46297582010329236, -0.35423422185228554, -0.23064413811105267]
    printOneTest(vt, 3, df, stream, antithetic, expected=expected1, context=f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [-0.46297582010329236, 0.35423422185228554, 0.23064413811105267]
    printOneTest(vt, 3, df, stream, antithetic, expected=expected2, context=f"seed {seed}, antithetic {antithetic}")

    # note: with seed 8675309 and stream 0, u1 and u2 should be the same in vunif, vt, ...
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u": u1, "x": expected1}
    printOneTest(vt, 3, df, stream, antithetic, as_dict, expected=expected, context=f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u": u2, "x": expected2}
    printOneTest(vt, 3, df, stream, antithetic, as_dict, expected=expected, context=f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vt_tests()