import numpy as np
from numpy.random import MT19937, Generator, SeedSequence
from scipy.stats import uniform
import subprocess

streams = [None,None,None]

epsilon = 1e-5 # for float "close enough"

def my_set_seed(seed: int) -> None:
    seed_seq = SeedSequence(seed)
    bit_generator = MT19937(seed_seq)
    for i in range(3):
        streams[i] = Generator(bit_generator)
        bit_generator = bit_generator.jumped()

def viaR(u: float | list[float], min_: float, max_: float, x_py: float) -> float:
    is_list = isinstance(u, list)
    if is_list: u = str(u).replace('[','c(').replace(']',')')
    x_via_r = f'Rscript -e "cat(qunif({u}, min = {min_}, max = {max_}), sep = \',\')"'
    result = subprocess.run(x_via_r, shell=True, capture_output=True, text=True).stdout
    if is_list:
        result = eval('[' + str(result) + ']', locals())
        is_corr = "✓" if all(abs(float(result[i]) - float(x_py[i])) <= epsilon \
               for i in range(len(result))) else "✗"
    else:
        result = float(result)
        is_corr = "✓" if abs(float(result) - x_py) <= epsilon else "✗"
    return result, is_corr

def print_u_vs_x(u: float | list[float], min_: float, max_: float, x: float | list[float], stream: int = 0) -> None:
    print(f"a = {min_}; b = {max_}; stream = {stream}")
    print(f"u = {u}")
    print(f"    x via py = {x}")
    x_r, is_corr = viaR(u, min_, max_, x)
    print(f"[{is_corr}] x via R  = {x_r}")
    print()


my_set_seed(8675309)
a = 0; b = 1
stream = 0
u = streams[stream].uniform(0,1)
x = uniform.ppf(u, loc = a, scale = b - a)
print_u_vs_x(u, a, b, x, stream)

my_set_seed(8675309)
a = 0; b = 1
stream = 0
u = [streams[stream].uniform(0,1) for i in range(5)]
x = uniform.ppf(u, loc = a, scale = b - a)
print_u_vs_x(u, a, b, x, stream)

my_set_seed(8675309)
a = 0; b = 1
stream = 0
u = [streams[stream].uniform(0,1) for i in range(10)]
x = uniform.ppf(u, loc = a, scale = b - a)
print_u_vs_x(u, a, b, x, stream)

my_set_seed(8675309)

a = 10; b = 20
stream = 0
u = streams[stream].uniform(0,1)
x = uniform.ppf(u, loc = a, scale = b - a)
print_u_vs_x(u, a, b, x, stream)

a = 13.2; b = 19.7
stream = 1
u = [streams[stream].uniform(0,1) for i in range(5)]
x = uniform.ppf(u, loc = a, scale = b - a)
print_u_vs_x(u, a, b, x, stream)

a = -3.1; b = 4.22
stream = 1  # same stream as before
u = [streams[stream].uniform(0,1) for i in range(3)]
x = uniform.ppf(u, loc = a, scale = b - a)
print_u_vs_x(u, a, b, x, stream)

# now let's test streams
my_set_seed(8675309)

a1 = 0; b1 = 10
u1 = [streams[0].uniform(0,1) for i in range(3)]
x1 = uniform.ppf(u1, loc = a1, scale = b1 - a1).tolist()
x1r, x1_is_corr = viaR(u1, a1, b1, x1)

a2 = -10; b2 = 0
u2 = [streams[1].uniform(0,1) for i in range(3)]
x2 = uniform.ppf(u2, loc = a2, scale = b2 - a2).tolist()
x2r, x2_is_corr = viaR(u2, a2, b2, x2)

a3 = -3; b3 = 3
u3 = [streams[2].uniform(0,1) for i in range(3)]
x3 = uniform.ppf(u3, loc = a3, scale = b3 - a3).tolist()
x3r, x3_is_corr = viaR(u3, a3, b3, x3)

print(f"For streams:")
print(f"a = {a1}; b = {b1}; stream = 0")
print(f"    x1 via py = {x1}")
print(f"[{x1_is_corr}] x1 via R  = {x1r}")
print(f"a = {a2}; b = {b2}; stream = 1")
print(f"    x2 via py = {x2}")
print(f"[{x2_is_corr}] x2 via R  = {x2r}")
print(f"a = {a3}; b = {b3}; stream = 1")
print(f"    x3 via py = {x3}")
print(f"[{x3_is_corr}] x3 via R  = {x3r}")
print()

# now let's test antithetic
a = -2.34; b = 8.61
my_set_seed(8675309)
stream = 0
u1 = [streams[stream].uniform(0,1) for i in range(3)]
x1 = uniform.ppf(u1, loc = a, scale = b - a).tolist()
print(f"For antithetic:")
print(f"a = {a}; b = {b}; stream = {stream}")
print(f"u1 = {u1}")
print(f"    x1 via py = {x1}")
x1r, is_corr = viaR(u1, a, b, x1)
print(f"[{is_corr}] x1 via R  = {x1r}")
print()

u2 = u1.copy()
for i in range(len(u2)): u2[i] = 1 - u2[i]
x2 = uniform.ppf(u2, loc = a, scale = b - a).tolist()
print(f"u2 = {u2}")
print(f"    x2 via py = {x2}")
x2r, is_corr = viaR(u2, a, b, x2)
print(f"[{is_corr}] x2 via R  = {x2r}")
print(f"u1 + u2 = {(np.array(u1) + np.array(u2)).tolist()}")
