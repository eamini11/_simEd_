# Import necessary libraries
# - numpy for array operations
import numpy as np
# - numpy.random for MT19937, Generator, SeedSequence
from numpy.random import MT19937, Generator, SeedSequence
# - scipy.stats.norm for norm.ppf (inverse normal CDF)
from scipy.stats import norm
# - subprocess for calling R validation
import subprocess

# Initialize stream containers and epsilon for float comparison
streams = [None, None, None]
# epsilon = 1e-5  # Tolerance for "close enough" float comparison
epsilon = 1e-5
# Function: my_set_seed(seed: int) -> None
def my_set_seed(seed: int) -> None:
    """Initialize multiple independent random streams using jumped MT19937"""
#   1. Create SeedSequence from input seed
    seed_seq = SeedSequence(seed)
#   2. Initialize MT19937 bit generator with seed sequence
    bit_generator = MT19937(seed_seq)
#   3. For each stream (0 to 2):
    for _ in range(3):
        streams[_] = Generator(bit_generator)
#      - Create Generator from current bit generator state
        bit_generator = bit_generator.jumped()
#      - Jump bit generator to next independent stream



# Function: viaR(u, mean, sd, x_py) -> tuple[result, correctness_indicator]
def viaR(
        u: float | list[float], 
        mean: float, sd: float, 
        x_py: float | list[float])-> tuple[float | list[float], str]:
# Purpose: Validate Python normal variates against R's qnorm function
    """
    Validate Python normal variates against R's qnorm function
    
    Args:
        u: Uniform random number(s) [0,1]
        mean: Normal distribution mean parameter
        sd: Normal distribution standard deviation parameter
        x_py: Python-generated normal variates to validate
    
    Returns:
        tuple: (R result, correctness indicator "✓" or "✗")
    """
# Steps:
#   1. Check if u is list or single value
    is_list = isinstance(u, list)
#   2. Format u for R: convert Python list to R vector syntax c(...)
    if is_list:
        u_r = str(u).replace('[', 'c(').replace(']', ')')
    else:
        u_r = str(u)
    
    # Call R's qnorm function via subprocess
    r_command = f'Rscript -e "cat(qnorm({u_r}, mean = {mean}, sd = {sd}), sep = \',\')"'
    result = subprocess.run(r_command, shell=True, capture_output=True, text=True).stdout
    #   5. Parse R output back to Python format
    if is_list:
         result = eval('[' + str(result) + ']', locals())
         is_corr = "✓" if all(abs(float(result[i]) - float(x_py[i])) <= epsilon 
                            for i in range(len(result))) else "✗"
    
    else:
        result = float(result)
        is_corr = "✓" if abs(float(result) - x_py) <= epsilon else "✗"
    
    return result, is_corr

# Function: print_u_vs_x(u, mean, sd, x, stream) -> None
def print_u_vs_x(u: float | list[float], 
                 mean: float, 
                 sd: float, 
                 x: float | list[float],
                 stream: int = 0)-> None:
# Purpose: Print formatted comparison of Python vs R normal generation
# Steps:
#   1. Print parameters: mean, sd, stream
    print(f"mean = {mean}, sd = {sd}, stream = {stream}")
#   2. Print uniform input: u
    print(f"u = {u}")
#   3. Print Python result: x
    print(f"x via py = {x}")
#   4. Call viaR to get R result and correctness
    x_r, is_corr = viaR(u, mean, sd, x)
#   5. Print R result with correctness indicator
    print(f"[{is_corr}] x via R = {x_r}")
    print()


print("=== Test Case 1: Single variate, mean=0, sd=1 ===")
my_set_seed(8675309)
mean, sd = 0.0, 0.1
stream = 0
u = streams[stream].uniform(0, 1)
x = norm.ppf(u, loc=mean, scale=sd)
print_u_vs_x(u, mean, sd, x, stream)


print("=== Test Case 2: Multiple variates, mean = 0, sd = 1 ===")
my_set_seed(8675309)
mean, sd = 0.0, 0.1
stream = 0
u = [streams[stream].uniform(0, 1) for i in range(5)]
x = norm.ppf(u, loc = mean, scale = sd).tolist()
print_u_vs_x(u, mean, sd, x, stream)


my_set_seed(8675309)
mean, sd = 0, 1
stream = 1
u = [streams[stream].uniform(0, 1) for i in range(5)]
x = norm.ppf(u, loc = mean, scale = sd).tolist()
print_u_vs_x(u, mean, sd, x, stream)



my_set_seed(8675309)
mean, sd = 0, 1
stream = 0
u = [streams[stream].uniform(0, 1) for i in range(2)]
x = norm.ppf(u, loc = mean, scale = sd).tolist()
print_u_vs_x(u, mean, sd, x, stream)


# Test Case 3: Custom mean and sd parameters
print("=== Test Case 3: Custom mean ")
my_set_seed(8675309)
mean, sd = 0.0, 0.1
stream = 0
u = [streams[stream].uniform(0, 1) for i in range(3)]
x = norm.ppf(u, loc = mean, scale = sd).tolist()
print_u_vs_x(u, mean, sd, x, stream)


print("=== Test Case 4: Different Streams ===")
my_set_seed(8675309)
mean1, sd1 = 0.0, 1.0
u1 = [streams[0].uniform(0, 1) for i in range(3)]
x1 = norm.ppf(u1, loc = mean1, scale = sd1).tolist()
x1r, x1_is_corr = viaR(u1, mean1, sd1, x1)

mean2, sd2 = 1, 0.5
u2 = [streams[1].uniform(0, 1) for i in range(3)]
x2 = norm.ppf(u2, loc=mean2, scale=sd2).tolist()
x2r, x2_is_corr = viaR(u2, mean2, sd2, x2)

mean3, sd3 = -1.0, 2
u3 = [streams[2].uniform(0, 1) for i in range(3)]
x3 = norm.ppf(u3, loc=mean3, scale=sd3).tolist()
x3r, x3_is_corr = viaR(u3, mean3, sd3, x3)
input()


print(f"For streams:")
print(f"mean = {mean1}, sd = {sd1}; stream = 0")
print(f" x1 via py = {x1}")
print(f"[{x1_is_corr}] x1 via R = {x1r}")
print(f"mean = {mean2}, sd = {sd2}; stream = 1")
print(f" x2 via py = {x2}")
print(f"[{x2_is_corr}] x2 via R = {x2r}")
print(f"mean = {mean3}, sd = {sd3}; stream = 2")
print(f" x3 via py = {x3}")
print(f"[{x3_is_corr}] x3 via R = {x3r}")
print()

print("=== Test Case 5: Antithetic variates ===")
mean, sd = 2.5, 0.8
my_set_seed(8675309)

# Generate original variates
u1 = [streams[0].uniform(0, 1) for i in range(3)]
x1 = norm.ppf(u1, loc=mean, scale=sd).tolist()
print(f"For antithetic:")
print(f"mean = {mean}, sd = {sd}; stream = 0")
print(f"u1 = {u1}")
print(f" x1 via py = {x1}")
x1r, is_corr = viaR(u1, mean, sd, x1)
print(f"[{is_corr}] x1 via R = {x1r}")
print()
# Generate antithetic variates: u2 = 1 - u1
u2 = [1 - u for u in u1]
x2 = norm.ppf(u2, loc=mean, scale=sd).tolist()

print(f"u2 = {u2}")
print(f" x2 via py = {x2}")
x2r, is_corr = viaR(u2, mean, sd, x2)
print(f"[{is_corr}] x2 via R = {x2r}")
print(f"u1 + u2 = {(np.array(u1) + np.array(u2)).tolist()}")  # Should be [1,1,1]
print()
input()

print("=== Test Case 6: Dictionary return format ===")
my_set_seed(8675309)
mean, sd = 0.0, 1.0
stream = 0
u = [streams[stream].uniform(0, 1) for i in range(3)]
x = norm.ppf(u, loc=mean, scale=sd).tolist()

# Simulate as_dict=True return format
result_dict = {"u": u, "x": x}
print(f"mean = {mean}, sd = {sd}; stream = {stream}")
print(f"as_dict = True")
print(f"result = {result_dict}")

# Validate the x values
x_r, is_corr = viaR(u, mean, sd, x)
print(f"[{is_corr}] x via R = {x_r}")
print()

# Test Case 7: Edge case - single variate with custom parameters
print("=== Test Case 7: Single variate with custom parameters ===")
my_set_seed(8675309)
mean, sd = 10.0, 3.0
stream = 1
u = streams[stream].uniform(0, 1)
x = norm.ppf(u, loc=mean, scale=sd)
print_u_vs_x(u, mean, sd, x, stream)

print("All tests completed!")