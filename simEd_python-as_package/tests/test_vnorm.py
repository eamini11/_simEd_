from simEd import set_seed, vunif
from simEd.generators.generators_continuous import vnorm
from typing import Any

from utils.printOneTest import printOneTest, Tests

################################################################################
def vnorm_tests() -> None:

    #############################################################
    # test exceptions
    try: print(vnorm(0, mean = 0, sd = 1))
    except Exception as err: print(f"correctly caught n error: {err}")

    try: print(vnorm(1, mean = 0, sd = 0))
    except Exception as err: print(f"correctly caught sd error: {err}")

    try: print(vnorm(1, mean = 0, sd = 1, stream = 88888))
    except Exception as err: print(f"correctly caught stream error: {err}")

    print('*' * 70)

    #############################################################
    # test no initialize streams
    n = 1; mean = 0; sd = 1; expected = Any
    printOneTest(vnorm, n, mean, sd, expected = expected, context = f"with no initial seed")

    seed = 8675309
    set_seed(seed)
    n = 1; mean = 0; sd = 1; stream = 0; expected =  0.42965188655973485
    printOneTest(vnorm, n, mean, sd, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; mean = 0; sd = 1; stream = 0
    expected = [0.42965188655973485, -0.33046542847494376, -0.2161028055678174, -1.3097223903414563, 0.8656442125577828]
    printOneTest(vnorm, n, mean, sd, stream, expected = expected, context = f"with initial seed {seed}")

    set_seed(seed)
    n = 5; mean = 0; sd = 1; stream = 0; antithetic = False; as_dict = True
    u = vunif(n, 0, 1)
    set_seed(seed)
    expected = [0.42965188655973485, -0.33046542847494376, -0.2161028055678174, -1.3097223903414563, 0.8656442125577828]
    expected = {"u":u, "x":expected}
    printOneTest(vnorm, n, mean, sd, stream, antithetic, as_dict, expected = expected, context = f"with initial seed {seed}")

    print('*' * 70)

    # test initial seed rng
    seed = 8675309
    set_seed(seed)
    n = 1; mean = 0; sd = 1
    expected = 0.42965188655973485
    printOneTest(vnorm, n, mean, sd, expected = expected, context = f"with initial seed {seed}, default stream 0")
    
    stream = 1
    n = 5; mean = 0; sd = 1
    expected = [-0.29375799847803075, 0.44876682292084985, -0.3341168288117192, -0.4816370675977582, -2.534269721583129]
    printOneTest(vnorm, n, mean, sd, stream, expected = expected, context = f"with initial seed {seed}, stream {stream}")
    ##########################################
    ##########################################
    # now let's test streams

    print('*' * 70)

    seed = 8675309
    set_seed(seed)
    n = 3; mean = 0.0; sd = 1.0; stream = 0
    expected = [0.42965188655973485, -0.33046542847494376, -0.2161028055678174]
    printOneTest(vnorm, n, mean, sd, stream, expected = expected, context = f"seed {seed}, stream {stream}")
    
    n = 3; mean = 1; sd = 0.5; stream = 1
    expected = [0.8531210007609846, 1.2243834114604248, 0.8329415855941404]
    printOneTest(vnorm, n, mean, sd, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    n = 3; mean = -1; sd = 2; stream = 2
    expected = [-0.7536469306764105, 1.0625927098930315, 1.3846165960234593]
    printOneTest(vnorm, n, mean, sd, stream, expected = expected, context = f"seed {seed}, stream {stream}")

    print('*' * 70)
    # and we should be able to produce the same numbers as before, but interlaced
    expected = [[0.42965188655973485, -0.33046542847494376, -0.2161028055678174],
                [0.8531210007609846, 1.2243834114604248, 0.8329415855941404],
                [-0.7536469306764105, 1.0625927098930315, 1.3846165960234593]]
    set_seed(seed)
    for i in range(9):
        stream = i % 3
        mean = 0 if stream == 0 else 1 if stream == 1 else -1
        sd = 1 if stream == 0 else 0.5 if stream == 1 else 2
        printOneTest(vnorm, 1, mean, sd, stream, expected = expected[stream][i // 3])
    
    ##########################################
    ##########################################
    # now let's test antithetic variates

    print('*' * 70)
    seed = 8675309
    mean = 2.5; sd = 0.8; stream = 0

    set_seed(seed)
    antithetic = False
    expected1 = [2.843721509247788, 2.235627657220045, 2.3271177555457463]
    printOneTest(vnorm, 3, mean, sd, stream, antithetic, expected = expected1, context = f"seed {seed}, antithetic {antithetic}")

    set_seed(seed)
    antithetic = True
    expected2 = [2.156278490752212, 2.764372342779955, 2.6728822444542537]
    printOneTest(vnorm, 3, mean, sd, stream, antithetic, expected = expected2, context = f"seed {seed}, antithetic {antithetic}")
    
    u1 = [0.6662755563376126, 0.37052415537598604, 0.4144538064289558]
    u2 = [0.3337244436623874, 0.629475844624014, 0.5855461935710442]

    set_seed(seed)
    antithetic = False; as_dict = True
    expected = {"u":u1, "x":expected1}
    printOneTest(vnorm, 3, mean, sd, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    set_seed(seed)
    antithetic = True; as_dict = True
    expected = {"u":u2, "x":expected2}
    printOneTest(vnorm, 3, mean, sd, stream, antithetic, as_dict, expected = expected, context = f"seed {seed}, antithetic {antithetic}, as_dict {as_dict}")

    ##########################################
    print('*' * 70)
    print(f"TOTAL TESTS:  {Tests.total}")
    print(f"TESTS FAILED: {Tests.incorrect}")

if __name__ == "__main__":
    vnorm_tests()