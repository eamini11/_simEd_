import numpy as np
from numpy.random import MT19937, Generator, SeedSequence
from scipy.stats import poisson
import subprocess

streams = [None,None,None]
epsilon = 1e-5 # for float "close enough"

def my_set_seed(seed: int) -> None:
    seed_seq = SeedSequence(seed)
    bit_generator = MT19937(seed_seq)
    for i in range(3):
        streams[i] = Generator(bit_generator)
        bit_generator = bit_generator.jumped()

def viaR(u: float | list[float], mu: float, x_py: int | list[int]) -> tuple:
    is_list = isinstance(u, list)
    if is_list: u = str(u).replace('[','c(').replace(']',')')
    x_via_r = f'Rscript -e "cat(qpois({u}, lambda = {mu}), sep = \',\')"'
    result = subprocess.run(x_via_r, shell=True, capture_output=True, text=True).stdout
    if is_list:
        result = eval('[' + str(result) + ']', locals())
        result = [int(x) for x in result]  # ensure integers
        is_corr = "✓" if all(int(result[i]) == int(x_py[i]) for i in range(len(result))) else "✗"
    else:
        result = int(float(result))
        is_corr = "✓" if int(result) == int(x_py) else "✗"
    return result, is_corr

def print_u_vs_x(u: float | list[float], mu: float, x: int | list[int], stream: int = 0) -> None:
    print(f"mu = {mu}; stream = {stream}")
    print(f"u = {u}")
    print(f" x via py = {x}")
    x_r, is_corr = viaR(u, mu, x)
    print(f"[{is_corr}] x via R = {x_r}")
    print()

my_set_seed(8675309)
mu = 2.0
stream = 0
u = streams[stream].uniform(0,1)
x = int(poisson.ppf(u, mu = mu))
print_u_vs_x(u, mu, x, stream)

my_set_seed(8675309)
mu = 2.0
stream = 0
u = [streams[stream].uniform(0,1) for i in range(5)]
x = [int(poisson.ppf(ui, mu = mu)) for ui in u]
print_u_vs_x(u, mu, x, stream)

my_set_seed(8675309)
mu = 2.0
stream = 0
u = [streams[stream].uniform(0,1) for i in range(10)]
x = [int(poisson.ppf(ui, mu = mu)) for ui in u]
print_u_vs_x(u, mu, x, stream)

my_set_seed(8675309)
mu = 1.5
stream = 0
u = streams[stream].uniform(0,1)
x = int(poisson.ppf(u, mu = mu))
print_u_vs_x(u, mu, x, stream)

mu = 3.0
stream = 1
u = [streams[stream].uniform(0,1) for i in range(5)]
x = [int(poisson.ppf(ui, mu = mu)) for ui in u]
print_u_vs_x(u, mu, x, stream)

mu = 0.5
stream = 1 # same stream as before
u = [streams[stream].uniform(0,1) for i in range(3)]
x = [int(poisson.ppf(ui, mu = mu)) for ui in u]
print_u_vs_x(u, mu, x, stream)

# now let's test streams
my_set_seed(8675309)
mu1 = 2.0
u1 = [streams[0].uniform(0,1) for i in range(3)]
x1 = [int(poisson.ppf(ui, mu = mu1)) for ui in u1]
x1r, x1_is_corr = viaR(u1, mu1, x1)

mu2 = 2.0
u2 = [streams[1].uniform(0,1) for i in range(3)]
x2 = [int(poisson.ppf(ui, mu = mu2)) for ui in u2]
x2r, x2_is_corr = viaR(u2, mu2, x2)


mu3 = 2.0
u3 = [streams[2].uniform(0,1) for i in range(3)]
x3 = [int(poisson.ppf(ui, mu = mu3)) for ui in u3]
x3r, x3_is_corr = viaR(u3, mu3, x3)

print(f"For streams:")
print(f"mu = {mu1}; stream = 0")
print(f" x1 via py = {x1}")
print(f"[{x1_is_corr}] x1 via R = {x1r}")
print(f"mu = {mu2}; stream = 1")
print(f" x2 via py = {x2}")
print(f"[{x2_is_corr}] x2 via R = {x2r}")
print(f"mu = {mu3}; stream = 2")
print(f" x3 via py = {x3}")
print(f"[{x3_is_corr}] x3 via R = {x3r}")
print()



# now let's test antithetic
mu = 2.5
my_set_seed(8675309)
u1 = [streams[0].uniform(0,1) for i in range(3)]
x1 = [int(poisson.ppf(ui, mu = mu)) for ui in u1]
print(f"For antithetic:")
print(f"mu = {mu}; stream = 0")
print(f"u1 = {u1}")
print(f" x1 via py = {x1}")
x1r, is_corr = viaR(u1, mu, x1)
print(f"[{is_corr}] x1 via R = {x1r}")
print()

u2 = [1 - ui for ui in u1]
x2 = [int(poisson.ppf(ui, mu=mu)) for ui in u2]

print(f"u2 (antithetic) = {u2}")
print(f"x2 via Python = {x2}")
x2r, is_corr2 = viaR(u2, mu, x2)
print(f"[{is_corr2}] x2 via R = {x2r}")