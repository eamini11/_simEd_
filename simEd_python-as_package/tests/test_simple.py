from simEd import *

################################################################################
def simple() -> None:
    for i in range(3): print(vunif(1, 2.2, 3.3), end = " ")
    print()
    for i in range(3): print(vexp(1, 2), end = " ")
    print()
    for i in range(3): print(vbinom(1, 10, 0.25), end = " ")
    print()

if __name__ == "__main__":
    simple()
