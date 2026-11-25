# Import necessary libraries
import numpy as np
from numpy.random import MT19937, Generator, SeedSequence
from scipy.stats import cauchy
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

def viaR(u: float | list[float], location: float, scale: float, x_py: float | list[float]) -> tuple[float | list[float], str]:
    """Validate Python Cauchy variates against R's qcauchy function"""
    is_list = isinstance(u, list)
    if is_list:
        u_r = str(u).replace('[', 'c(').replace(']', ')')
    else:
        u_r = str(u)
    
    r_command = f'Rscript -e "cat(qcauchy({u_r}, location = {location}, scale = {scale}), sep = \',\')"'
    result = subprocess.run(r_command, shell=True, capture_output=True, text=True).stdout
    
    if is_list:
        result = eval('[' + str(result) + ']', locals())
        is_corr = "✓" if all(abs(float(result[i]) - float(x_py[i])) <= epsilon 
                            for i in range(len(result))) else "✗"
    else:
        result = float(result)
        is_corr = "✓" if abs(float(result) - x_py) <= epsilon else "✗"
    
    return result, is_corr

def print_u_vs_x(u: float | list[float], location: float, scale: float, x: float | list[float], stream: int = 0) -> None:
    """Print formatted comparison of Python vs R Cauchy generation"""
    print(f"location = {location}, scale = {scale}, stream = {stream}")
    print(f"u = {u}")
    print(f"x via py = {x}")
    x_r, is_corr = viaR(u, location, scale, x)
    print(f"[{is_corr}] x via R = {x_r}")
    print()

print("=== Test Case 1: Single variate, location=0, scale=1 ===")
my_set_seed(8675309)
location = 0
scale = 1
stream = 0
u = [streams[stream].uniform(0, 1) for i in range(5)]
x = cauchy.ppf(u, loc=location, scale=scale)
print_u_vs_x(u, location, scale, x, stream)


my_set_seed(8675309)
location = 0.0
scale = 1.0
stream = 0
u = streams[stream].uniform(0, 1) 
x = cauchy.ppf(u, loc=location, scale=scale)
print_u_vs_x(u, location, scale, x, stream)


print("=== Test Case 2: Different Streams")
my_set_seed(8675309)
location = 0.0
scale = 1.0
stream = 0
u = [streams[stream].uniform(0, 1) for i in range(3)]
x = cauchy.ppf(u, loc=location, scale=scale).tolist()
print_u_vs_x(u, location, scale, x, stream)

location = 1.0
scale = 2.0
stream = 1
u = [streams[stream].uniform(0, 1) for i in range(3)]
x = cauchy.ppf(u, loc=location, scale=scale).tolist()
print_u_vs_x(u, location, scale, x, stream)


location = -1.0
scale = 0.5
stream = 2
u = [streams[stream].uniform(0, 1) for i in range(3)]
x = cauchy.ppf(u, loc=location, scale=scale).tolist()
print_u_vs_x(u, location, scale, x, stream)

print("=== Test Case 4: Antithetic variates ===")
my_set_seed(8675309)
location = 0.0
scale = 1.0
stream = 0
u1 = [streams[0].uniform(0, 1) for i in range(3)]
x1 = cauchy.ppf(u1, loc=location, scale=scale).tolist()
print(f"For antithetic:")
print(f"location = {location}, scale = {scale}; stream = 0")
print(f"u1 = {u1}")
print(f" x1 via py = {x1}")
x1r, is_corr = viaR(u1, location, scale, x1)
print(f"[{is_corr}] x1 via R = {x1r}")
print()


u2 = [1 - u for u in u1]
x2 = cauchy.ppf(u2, loc=location, scale=scale).tolist()
print(f"u2 = {u2}")
print(f" x2 via py = {x2}")
x2r, is_corr = viaR(u2, location, scale, x2)
print(f"[{is_corr}] x2 via R = {x2r}")
print(f"u1 + u2 = {(np.array(u1) + np.array(u2)).tolist()}")
print()
input()
print("All Cauchy tests completed!")