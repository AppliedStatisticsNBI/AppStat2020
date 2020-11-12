#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:31:47 2018

@author: michelsen
"""

import numpy as np
import pandas as pd


def gen_data(N=50, sigma_x=1, sigma_y=8):

    # set random seed
    np.random.seed(42)

    # define ranges for the x value (exercise)
    ranges = [(0, 10), (2, 12), (5, 15), (8, 18)]
    # slopes for the different groups
    slopes = [-1.5, -1.9, -2.3, -2.7]
    # offsets for the different groups
    offsets = [20, 45, 70, 95]
    # ages for the different groups
    ages = ["10-20", "20-40", "40-60", "60-80"]
    # number of groups
    N_groups = len(ranges)

    # make the x-axis for the different groups with the specified ranges,
    # adds random, Gaussian noise to the data and truncate negative values
    xs = np.array([np.linspace(xmin, xmax, N) for (xmin, xmax) in ranges])
    xs += np.random.normal(0, sigma_x, (N_groups, N))
    xs[xs < 0] = 0

    # calculate the y values for the different groups by the linear function
    # and adds random, Gaussian noise to the data and truncate values less than 10
    ys = np.array([a * x + b for (a, b, x) in zip(slopes, offsets, xs)])
    ys += np.random.normal(0, sigma_y, (N_groups, N))
    ys[ys < 0] = 10

    # combine all the data into a dictionary
    data = {
        "Exercise": xs.flatten(),
        "Age": np.repeat(ages, N),
        "Probability": ys.flatten(),
    }

    # convert the dictionary to a Pandas dataframe
    df = pd.DataFrame(data)
    # shuffle the rows to make the pattern in the data seem less visible
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # save the data to a csv file
    df.to_csv("Simpsons_paradox.csv")

    return df


gen_data()