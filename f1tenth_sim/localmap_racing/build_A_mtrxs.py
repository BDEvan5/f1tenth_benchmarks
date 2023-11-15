import numpy as np 
import math

def build_A(path_length):
    no_splines = path_length - 1

    M = np.zeros((no_splines * 4, no_splines * 4))
        # calculate scaling factors between every pair of splines

    scaling = np.ones(no_splines - 1)
    template_M = np.array(                          # current point               | next point              | bounds
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                 [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                 [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1i+1             = 0
                 [0,  0,  2,  6,  0,  0, -2,  0]])  # _             2a_2i + 6a_3i               - 2a_2i+1   = 0

    for i in range(no_splines):
        j = i * 4

        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M

            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)

        else:
            # no curvature and heading bounds on last element (handled afterwards)
            M[j: j + 2, j: j + 4] = [[1,  0,  0,  0],
                                     [1,  1,  1,  1]]

    # if the path is unclosed we want to fix heading at the start and end point of the path (curvature cannot be
    # determined in this case) -> set heading boundary conditions

    # heading start point
    M[-2, 1] = 1  # heading start point (evaluated at t = 0)

    # heading end point
    M[-1, -4:] = [0, 1, 2, 3]  # heading end point (evaluated at t = 1)

    return M


import os
def build_A_matrixes():
    start = 10
    end = 52

    path = f"Logs/Data_A/"
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(start, end+1):
        print(f"{i} is being built")
        A = build_A(i)

        inv = np.linalg.inv(A)
        np.save(path + f"A_{i}.npy", inv)


if __name__ == "__main__":
    build_A_matrixes()



