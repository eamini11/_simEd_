# Import necessary libraries
import numpy as np
from numpy.random import MT19937, Generator
from simEd import set_seed
from simEd.generators.generators_continuous import vgamma
from scipy.stats import gamma
import subprocess

# Initialize stream containers and epsilon for float comparison
streams = [None, None, None]
epsilon = 1e-5

def my_set_seed(seed: int) -> None:
    """Initialize multiple independent random streams using jumped MT19937"""
    seed_seq = set_seed(seed)
    bit_generator = MT19937(seed_seq)
    for i in range(3):
        streams[i] = Generator(bit_generator)
        bit_generator = bit_generator.jumped()

def viaR(u: float | list[float], shape: float, rate: float, x_py: float | list[float]) -> tuple[float | list[float], str]:
    """Validate Python gamma variates against R's qgamma function"""
    is_list = isinstance(u, list)
    if is_list:
        u_r = str(u).replace('[', 'c(').replace(']', ')')
    else:
        u_r = str(u)
    
    r_command = f'Rscript -e "cat(qgamma({u_r}, shape = {shape}, rate = {rate}), sep = \',\')"'
    result = subprocess.run(r_command, shell=True, capture_output=True, text=True).stdout
    
    if is_list:
        result = eval('[' + str(result) + ']', locals())
        is_corr = "✓" if all(abs(float(result[i]) - float(x_py[i])) <= epsilon 
                           for i in range(len(result))) else "✗"
    else:
        result = float(result)
        is_corr = "✓" if abs(float(result) - x_py) <= epsilon else "✗"
    
    return result, is_corr

def print_u_vs_x(u: float | list[float], shape: float, rate: float, x: float | list[float], stream: int = 0) -> None:
    """Print formatted comparison of Python vs R gamma generation"""
    print(f"shape = {shape}, rate = {rate}, stream = {stream}")
    print(f"u = {u}")
    print(f"x via py = {x}")
    x_r, is_corr = viaR(u, shape, rate, x)
    print(f"[{is_corr}] x via R = {x_r}")
    print()

print("=== Test Case 1: Single variate, shape=2, rate=1 ===")
my_set_seed(8675309)
shape, rate = 2.0, 1.0
stream = 0
u = streams[stream].uniform(0, 1)
x = gamma.ppf(u, a=shape, scale=1/rate)
print_u_vs_x(u, shape, rate, x, stream)


print("=== Test Case 2: Multiple variates, shape=2, rate=0.5 ===")
set_seed(8675309)
shape, rate = 2.0, 0.5
stream = 0
u = [streams[stream].uniform(0, 1) for i in range(5)]
x = vgamma(5, shape, rate, stream)
print_u_vs_x(u, shape, rate, x, stream)



print("=== Test Case 3: Different stream ===")
set_seed(8675309)
# Stream 0: shape=2, rate=1
shape1, rate1 = 2.0, 1.0
u1 = [streams[0].uniform(0, 1) for i in range(3)]
x1 = gamma.ppf(u1, a=shape1, scale=1/rate1).tolist()
x1r, x1_is_corr = viaR(u1, shape1, rate1, x1)

# Stream 1: shape=1.0, rate=2.0  
shape2, rate2 = 1.0, 2.0
u2 = [streams[1].uniform(0, 1) for i in range(3)]
x2 = gamma.ppf(u2, a=shape2, scale=1/rate2).tolist()
x2r, x2_is_corr = viaR(u2, shape2, rate2, x2)

# Stream 2: shape=0.5, rate=0.5
shape3, rate3 = 0.5, 0.5
u3 = [streams[2].uniform(0, 1) for i in range(3)]
x3 = gamma.ppf(u3, a=shape3, scale=1/rate3).tolist()
x3r, x3_is_corr = viaR(u3, shape3, rate3, x3)

# Print results
print(f"For streams:")
print(f"shape = {shape1}, rate = {rate1}; stream = 0")
print(f" x1 via py = {x1}")
print(f"[{x1_is_corr}] x1 via R = {x1r}")
print(f"shape = {shape2}, rate = {rate2}; stream = 1") 
print(f" x2 via py = {x2}")
print(f"[{x2_is_corr}] x2 via R = {x2r}")
print(f"shape = {shape3}, rate = {rate3}; stream = 2")
print(f" x3 via py = {x3}")
print(f"[{x3_is_corr}] x3 via R = {x3r}")

print("=== Test Case 4: Custom shape and rate parameters ===")
my_set_seed(8675309)
shape, rate = 3.0, 2.0
stream = 0
u = [streams[stream].uniform(0, 1) for i in range(3)]
x = gamma.ppf(u, a=shape, scale=1/rate).tolist()
print_u_vs_x(u, shape, rate, x, stream)

print("=== Test Case 5: Multiple Streams ===")
set_seed(8675309)
shape1, rate1 = 2.0, 1.0
u1 = [streams[0].uniform(0, 1) for i in range(3)]
x1 = vgamma(3, shape1, rate1, stream)
x1r, x1_is_corr = viaR(u1, shape1, rate1, x1)

shape2, rate2 = 1.0, 2.0
u2 = [streams[1].uniform(0, 1) for i in range(3)]
x2 = vgamma(3, shape2, rate2, stream = 1)
x2r, x2_is_corr = viaR(u2, shape2, rate2, x2)

shape3, rate3 = 0.5, 0.5
u3 = [streams[2].uniform(0, 1) for i in range(3)]
x3 = vgamma(3, shape3, rate3, stream = 2)
x3r, x3_is_corr = viaR(u3, shape3, rate3, x3)

print(f"For streams:")
print(f"shape = {shape1}, rate = {rate1}; stream = 0")
print(f" x1 via py = {x1}")
print(f"[{x1_is_corr}] x1 via R = {x1r}")
print(f"shape = {shape2}, rate = {rate2}; stream = 1")
print(f" x2 via py = {x2}")
print(f"[{x2_is_corr}] x2 via R = {x2r}")
print(f"shape = {shape3}, rate = {rate3}; stream = 2")
print(f" x3 via py = {x3}")
print(f"[{x3_is_corr}] x3 via R = {x3r}")
print()


print("=== Test Case 6: Antithetic variates ===")
set_seed(8675309)
shape, rate = 1.5, 0.8

u1 = [streams[0].uniform(0, 1) for i in range(3)]
x1 = gamma.ppf(u1, a=shape, scale=1/rate).tolist()
print(f"For antithetic:")
print(f"shape = {shape}, rate = {rate}; stream = 0")
print(f"u1 = {u1}")
print(f" x1 via py = {x1}")
x1r, is_corr = viaR(u1, shape, rate, x1)
print(f"[{is_corr}] x1 via R = {x1r}")
print()
input()

u2 = [1 - u for u in u1]
x2 = gamma.ppf(u2, a=shape, scale=1/rate).tolist()
print(f"u2 = {u2}")
print(f" x2 via py = {x2}")
x2r, is_corr = viaR(u2, shape, rate, x2)
print(f"[{is_corr}] x2 via R = {x2r}")
print(f"u1 + u2 = {(np.array(u1) + np.array(u2)).tolist()}")
print()

print("All tests completed!")