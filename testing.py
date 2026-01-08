import multiprocessing as mp
import numpy as np
import os
from PIL import Image
from matplotlib import imshow

def parallel_test(x):
    print(f"x = {x}, PID = {os.getpid()}")
    return x * x


if __name__ == "__main__":
    foo = np.array([0, 1, 2, 3, 4, 5])

    # Optional but explicit on macOS:
    # mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=6) as pool:
        result = pool.map(parallel_test, foo)

    print("Result:", result)
# The tasks are not executed in order!
im = Image.open( './data/kings_cross.jpg')
im