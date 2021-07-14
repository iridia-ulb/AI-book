from random import randint
import numpy as np

def readSudokus(n, filename="sudokus.txt"):
    """
    Returns the n first sudokus of the file with given name
    """
    f = open(filename)
    res = []
    for s in range(n):
        txt = f.readline().strip()
        if txt != "":
            res.append([[int(txt[i+j*9]) for i in range(9)] for j in range(9)])
    f.close()
    return np.array(res)
