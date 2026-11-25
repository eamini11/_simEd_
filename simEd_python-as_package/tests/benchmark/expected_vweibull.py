# Import necessary libraries
import numpy as np
from numpy.random import MT19937, Generator, SeedSequence
from scipy.stats import weibull_min
import subprocess

# Initialize stream containers and epsilon for float comparison
streams = [None, None, None]
epsilon = 1e-5

def my_set_seed(seed: int) -> None:
    """Initialize multiple independent random streams using jumped MT19937"""
    seed_seq = SeedSequence(seed)
    bit_generator = MT19937(seed_seq)
    for i in range(3):
        streams[i] = Generator(bit_generator)
        bit_generator = bit_generator.jumped()

def viaR(u: float | list[float], shape: float, scale: float, x_py: float | list[float]) -> tuple[float | list[float], str]:
    """Validate Python Weibull variates against R's qweibull function"""
    is_list = isinstance(u, list)
    if is_list:
        u_r = str(u).replace('[', 'c(').replace(']', ')')
    else:
        u_r = str(u)
    
    r_command = f'Rscript -e "cat(qweibull({u_r}, shape = {shape}, scale = {scale}), sep = \',\')"'
    result = subprocess.run(r_command, shell=True, capture_output=True, text=True).stdout
    
    if is_list:
        result = eval('[' + str(result) + ']', locals())
        is_corr = "✓" if all(abs(float(result[i]) - float(x_py[i])) <= epsilon 
                            for i in range(len(result))) else "✗"
    else:
        result = float(result)
        is_corr = "✓" if abs(float(result) - x_py) <= epsilon else "✗"
    
    return result, is_corr

def print_u_vs_x(u: float | list[float], shape: float, scale: float, x: float | list[float], stream: int = 0) -> None:
    """Print formatted comparison of Python vs R Weibull generation"""
    print(f"shape = {shape}, scale = {scale}, stream = {stream}")
    print(f"u = {u}")
    print(f"x via py = {x}")
    x_r, is_corr = viaR(u, shape, scale, x)
    print(f"[{is_corr}] x via R = {x_r}")
    print()

print("=== Test Case 1: Single variate, shape=2, scale=1 ===")
my_set_seed(8675309)
shape = 2
scale = 1
stream = 0
u = streams[stream].uniform(0, 1)
x = weibull_min.ppf(u, c=shape, scale=scale)
print_u_vs_x(u, shape, scale, x, stream)


my_set_seed(8675309)
shape = 2
scale = 1
stream = 0
u = [streams[stream].uniform(0, 1) for i in range(5)]
x = weibull_min.ppf(u, c=shape, scale=scale)
print_u_vs_x(u, shape, scale, x, stream)


my_set_seed(8675309)
shape = 2.0
scale = 1.0
stream = 0
antithetic = False
asDict = True
u = [streams[stream].uniform(0, 1) for i in range(5)]
x = weibull_min.ppf(u, c=shape, scale=scale)
print_u_vs_x(u, shape, scale, x, stream)


print("=== Test Case 2: Multiple variates, shape=2, scale=1 ===")
my_set_seed(8675309)
shape = 1.5
scale = 2.0
u = streams[stream].uniform(0, 1) 
x = weibull_min.ppf(u, c=shape, scale=scale).tolist()
print_u_vs_x(u, shape, scale, x)


shape = 1.5
scale = 2.0
stream = 1
u = [streams[stream].uniform(0, 1)for i in range(3)] 
x = weibull_min.ppf(u, c=shape, scale=scale).tolist()
print_u_vs_x(u, shape, scale, x, stream)


shape = 3.0
scale = 0.5
stream = 2
u = [streams[stream].uniform(0, 1)for i in range(3)] 
x = weibull_min.ppf(u, c=shape, scale=scale).tolist()
print_u_vs_x(u, shape, scale, x, stream)


print("=== Test Case 3: Antithetic variates ===")
my_set_seed(8675309)
shape = 2.0
scale = 1.0
stream = 0
u1 = [streams[0].uniform(0, 1) for i in range(3)]
x1 = weibull_min.ppf(u1, c=shape, scale=scale).tolist()
print(f"For antithetic:")
print(f"shape = {shape}, scale = {scale}; stream = 0")
print(f"u1 = {u1}")
print(f" x1 via py = {x1}")
x1r, is_corr = viaR(u1, shape, scale, x1)
print(f"[{is_corr}] x1 via R = {x1r}")
print()

u2 = [1 - u for u in u1]
x2 = weibull_min.ppf(u2, c=shape, scale=scale).tolist()
print(f"u2 = {u2}")
print(f" x2 via py = {x2}")
x2r, is_corr = viaR(u2, shape, scale, x2)
print(f"[{is_corr}] x2 via R = {x2r}")
print(f"u1 + u2 = {(np.array(u1) + np.array(u2)).tolist()}")
print()

input()
print("All Weibull tests completed!")