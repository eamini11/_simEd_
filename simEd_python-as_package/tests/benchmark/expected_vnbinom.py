import numpy as np
from numpy.random import MT19937, Generator, SeedSequence
from scipy.stats import nbinom
import subprocess

streams = [None,None,None]
epsilon = 1e-5 # for float "close enough"

def my_set_seed(seed: int) -> None:
    seed_seq = SeedSequence(seed)
    bit_generator = MT19937(seed_seq)
    for i in range(3):
        streams[i] = Generator(bit_generator)
        bit_generator = bit_generator.jumped()

def viaR(u: float | list[float], size: int, prob: float, x_py: int | list[int]) -> tuple:
    is_list = isinstance(u, list)
    if is_list: u = str(u).replace('[','c(').replace(']',')')
    x_via_r = f'Rscript -e "cat(qnbinom({u}, size = {size}, prob = {prob}), sep = \',\')"'
    result = subprocess.run(x_via_r, shell=True, capture_output=True, text=True).stdout
    if is_list:
        result = eval('[' + str(result) + ']', locals())
        result = [int(x) for x in result]  # ensure integers
        is_corr = "✓" if all(int(result[i]) == int(x_py[i]) for i in range(len(result))) else "✗"
    else:
        result = int(float(result))
        is_corr = "✓" if int(result) == int(x_py) else "✗"
    return result, is_corr

def print_u_vs_x(u: float | list[float], size: int, prob: float, x: int | list[int], stream: int = 0) -> None:
    print(f"size = {size}; prob = {prob}; stream = {stream}")
    print(f"u = {u}")
    print(f" x via py = {x}")
    x_r, is_corr = viaR(u, size, prob, x)
    print(f"[{is_corr}] x via R = {x_r}")
    print()

my_set_seed(8675309)
size = 5
prob = 0.5
stream = 0
u = streams[stream].uniform(0,1)
x = int(nbinom.ppf(u, n = size, p = prob))
print_u_vs_x(u, size, prob, x, stream)


my_set_seed(8675309)
size = 5
prob = 0.5
stream = 0
u = [streams[stream].uniform(0,1) for i in range(5)]
x = [int(nbinom.ppf(ui, n = size, p = prob)) for ui in u]
print_u_vs_x(u, size, prob, x, stream)


my_set_seed(8675309)
size = 5
prob = 0.5
stream = 0
u = [streams[stream].uniform(0,1) for i in range(10)]
x = [int(nbinom.ppf(ui, n = size, p = prob)) for ui in u]
print_u_vs_x(u, size, prob, x, stream)


my_set_seed(8675309)
size = 3
prob = 0.3
stream = 0
u = streams[stream].uniform(0,1)
x = int(nbinom.ppf(u, n = size, p = prob))
print_u_vs_x(u, size, prob, x, stream)


size = 4
prob = 0.4
stream = 1
u = [streams[stream].uniform(0,1) for i in range(5)]
x = [int(nbinom.ppf(ui, n = size, p = prob)) for ui in u]
print_u_vs_x(u, size, prob, x, stream)


size = 2
prob = 0.7
stream = 1 # same stream as before
u = [streams[stream].uniform(0,1) for i in range(3)]
x = [int(nbinom.ppf(ui, n = size, p = prob)) for ui in u]
print_u_vs_x(u, size, prob, x, stream)

# now let's test streams
my_set_seed(8675309)
size1 = 5
prob1 = 0.5
u1 = [streams[0].uniform(0,1) for i in range(3)]
x1 = [int(nbinom.ppf(ui, n = size1, p = prob1)) for ui in u1]
x1r, x1_is_corr = viaR(u1, size1, prob1, x1)


size2 = 5
prob2 = 0.5
u2 = [streams[1].uniform(0,1) for i in range(3)]
x2 = [int(nbinom.ppf(ui, n = size2, p = prob2)) for ui in u2]
x2r, x2_is_corr = viaR(u2, size2, prob2, x2)

size3 = 5
prob3 = 0.5
u3 = [streams[2].uniform(0,1) for i in range(3)]
x3 = [int(nbinom.ppf(ui, n = size3, p = prob3)) for ui in u3]
x3r, x3_is_corr = viaR(u3, size3, prob3, x3)

print(f"For streams:")
print(f"size = {size1}; prob = {prob1}; stream = 0")
print(f" x1 via py = {x1}")
print(f"[{x1_is_corr}] x1 via R = {x1r}")
print(f"size = {size2}; prob = {prob2}; stream = 1")
print(f" x2 via py = {x2}")
print(f"[{x2_is_corr}] x2 via R = {x2r}")
print(f"size = {size3}; prob = {prob3}; stream = 2")
print(f" x3 via py = {x3}")
print(f"[{x3_is_corr}] x3 via R = {x3r}")
print()

# now let's test antithetic
size = 6
prob = 0.6
my_set_seed(8675309)
u1 = [streams[0].uniform(0,1) for i in range(3)]
x1 = [int(nbinom.ppf(ui, n = size, p = prob)) for ui in u1]
print(f"For antithetic:")
print(f"size = {size}; prob = {prob}; stream = 0")
print(f"u1 = {u1}")
print(f" x1 via py = {x1}")
x1r, is_corr = viaR(u1, size, prob, x1)
print(f"[{is_corr}] x1 via R = {x1r}")
print()
input()

u2 = u1.copy()
for i in range(len(u2)): u2[i] = 1 - u2[i]
x2 = [int(nbinom.ppf(ui, n = size, p = prob)) for ui in u2]
print(f"u2 = {u2}")
print(f" x2 via py = {x2}")
x2r, is_corr = viaR(u2, size, prob, x2)
print(f"[{is_corr}] x2 via R = {x2r}")
print(f"u1 + u2 = {(np.array(u1) + np.array(u2)).tolist()}")