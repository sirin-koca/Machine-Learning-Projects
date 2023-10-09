#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read CSV
titanic = pd.read_csv('titanic.csv')

""" The function distplot from the Seaborn library is deprecated and will be removed in future versions. Seaborn now 
provides two main functions to replace distplot: displot (for figure-level plots) and histplot (for axes-level 
plots)."""
# Seaborn histogram plot
sns.histplot(titanic['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution using Seaborn')
plt.show()

# Boxplot comparing age distributions of those who survived vs those who didn't
sns.boxplot(x='Survived', y='Age', data=titanic)
plt.title('Age distribution based on Survival')
plt.show()


if __name__ == '__main__':
    # Your code here

    sys.exit()
