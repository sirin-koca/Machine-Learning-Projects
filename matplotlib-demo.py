#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import matplotlib.pyplot as plt

# In this demo we will use the titanic.csv dataset
# Distribution of Ages on Titanic

# Read CSV
titanic = pd.read_csv('titanic.csv')

# Simple plot
ages = titanic['Age'].dropna()  # Removing NaN values
plt.hist(ages, bins=30, edgecolor='black')
plt.xlabel('Age')  # Age on the x-axis
plt.ylabel('Count')  # Count on the y-axis
plt.title('Distribution of Ages on Titanic')
plt.show()

if __name__ == '__main__':
    # Your code here

    sys.exit()
