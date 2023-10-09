#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np

# Creating a numpy array
arr = np.array([1, 2, 3, 4, 5])

# Basic operations
print(arr + 10)  # Add 10 to each element
print(arr * 2)  # Multiply each element by 2

# Statistical operations
print(np.mean(arr))
print(np.median(arr))
print(np.std(arr))  # Standard deviation

if __name__ == '__main__':
    # Your code here

    sys.exit()
