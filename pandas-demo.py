#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd

# Read CSV file
titanic = pd.read_csv('titanic.csv')

# Display the first 5 rows
print(titanic.head())

# Basic operations
print(titanic['Age'].mean())     # Average age of passengers
print(titanic['Sex'].value_counts())  # Count of each unique value in 'Sex' column

# Filtering data
survived = titanic[titanic['Survived'] == 1]
print(survived.head())


if __name__ == '__main__':
    # Your code here

    sys.exit()
