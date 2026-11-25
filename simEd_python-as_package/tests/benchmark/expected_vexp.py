import numpy as np
from numpy.random import MT19937, Generator, SeedSequence
from scipy.stats import expon
import subprocess

streams = [None,None,None]

epsilon = 1e-5 # for float "close enough"

def my_set_seed(seed: int) -> None:
    seed_seq = SeedSequence(seed)
    bit_generator = MT19937(seed_seq)
    for i in range(3):
        streams[i] = Generator(bit_generator) 
        bit_generator = bit_generator.jumped()

def viaR(u: float | list[float], rate: float, x_py: float) -> float:
    is_list = isinstance(u, list)
    if is_list: u = str(u).replace('[','c(').replace(']',')')
    x_via_r = f'Rscript -e "cat(qexp({u}, rate = {rate}), sep = \',\')"'
    result = subprocess.run(x_via_r, shell=True, capture_output=True, text=True).stdout
    if is_list:
        result = eval('[' + str(result) + ']', locals())
        is_corr = "✓" if all(abs(float(result[i]) - float(x_py[i])) <= epsilon \
               for i in range(len(result))) else "✗"
    else:
        result = float(result)
        is_corr = "✓" if abs(float(result) - x_py) <= epsilon else "✗"
    return result, is_corr

def print_u_vs_x(u: float | list[float], rate: float, x: float | list[float], stream: int = 0) -> None:
    print(f"rate = {rate}; stream = {stream}")
    print(f"u = {u}")
    print(f"    x via py = {x}")
    x_r, is_corr = viaR(u, rate, x)
    print(f"[{is_corr}] x via R  = {x_r}")
    print()

my_set_seed(8675309)
rate = 1/2
stream = 0
u = streams[stream].uniform(0,1)
x = expon.ppf(u, loc = 0, scale = 1 / rate) 
print_u_vs_x(u, rate, x, stream)


my_set_seed(8675309)
rate = 1/2
stream = 0
u = [streams[stream].uniform(0,1) for i in range(5)]
x = expon.ppf(u, loc = 0, scale = 1 / rate).tolist()
print_u_vs_x(u, rate, x, stream)
input()

my_set_seed(8675309)
rate = 1/2
stream = 0
u = [streams[stream].uniform(0,1) for i in range(10)]
x = expon.ppf(u, loc = 0, scale = 1 / rate).tolist()
print_u_vs_x(u, rate, x, stream)


my_set_seed(8675309)
rate = 1.0
stream = 0
u = streams[stream].uniform(0,1)
x = expon.ppf(u, loc = 0, scale = 1 / rate)
print_u_vs_x(u, rate, x, stream)

rate = 1/2
stream = 1
u = [streams[stream].uniform(0,1) for i in range(5)]
x = expon.ppf(u, loc = 0, scale = 1 / rate).tolist()
print_u_vs_x(u, rate, x, stream)

rate = 2.0
stream = 1  # same stream as before
u = [streams[stream].uniform(0,1) for i in range(3)]
x = expon.ppf(u, loc = 0, scale = 1 / rate).tolist()
print_u_vs_x(u, rate, x, stream)


# now let's test streams
my_set_seed(8675309)

rate1 = 1/2
u1 = [streams[0].uniform(0,1) for i in range(3)]
x1 = expon.ppf(u1, loc = 0, scale = 1 / rate1).tolist()
x1r, x1_is_corr = viaR(u1, rate1, x1)

rate2 = 1
u2 = [streams[1].uniform(0,1) for i in range(3)]
x2 = expon.ppf(u2, loc = 0, scale = 1 / rate2).tolist()
x2r, x2_is_corr = viaR(u2, rate2, x2)

rate3 = 2
u3 = [streams[2].uniform(0,1) for i in range(3)]
x3 = expon.ppf(u3, loc = 0, scale = 1 / rate3).tolist()
x3r, x3_is_corr = viaR(u3, rate3, x3)

print(f"For streams:")
print(f"rate = {rate1}; stream = 0")
print(f"    x1 via py = {x1}")
print(f"[{x1_is_corr}] x1 via R  = {x1r}")
print(f"rate = {rate2}; stream = 1")
print(f"    x2 via py = {x2}")
print(f"[{x2_is_corr}] x2 via R  = {x2r}")
print(f"rate = {rate3}; stream = 2")
print(f"    x3 via py = {x3}")
print(f"[{x3_is_corr}] x3 via R  = {x3r}")
print()
input()
# now let's test antithetic
rate = 0.675
my_set_seed(8675309)
u1 = [streams[0].uniform(0,1) for i in range(3)]
x1 = expon.ppf(u1, loc = 0, scale = 1 / rate).tolist()
print(f"For antithetic:")
print(f"rate = {rate}; stream = 0")
print(f"u1 = {u1}")
print(f"    x1 via py = {x1}")
x1r, is_corr = viaR(u1, rate, x1)
print(f"[{is_corr}] x1 via R  = {x1r}")
print()

u2 = u1.copy()
for i in range(len(u2)): u2[i] = 1 - u2[i]
x2 = expon.ppf(u2, loc = 0, scale = 1 / rate).tolist()
print(f"u2 = {u2}")
print(f"    x2 via py = {x2}")
x2r, is_corr = viaR(u2, rate, x2)
print(f"[{is_corr}] x2 via R  = {x2r}")
print(f"u1 + u2 = {(np.array(u1) + np.array(u2)).tolist()}")
