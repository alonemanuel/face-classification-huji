import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def log(*args):
    print('Log: ', end='')
    for arg in args:
        print(arg, end='')
    print()