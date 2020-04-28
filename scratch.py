from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from hw3.plot import *

if __name__ == '__main__':
    # for dirname, _, filenames in os.walk('data/covid_reddit_posts'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))

    # ================ #

    nRowsRead = 1000  # specify 'None' if want to read whole file
    # coronavirus_reddit_clean_comments.csv has more rows, this is only loading/previewing the first 1000 rows
    df1 = pd.read_csv('data/covid_reddit_posts/coronavirus_reddit_clean_comments.csv', delimiter=',', nrows=nRowsRead)
    df1.dataframeName = 'coronavirus_reddit_clean_comments.csv'
    nRow, nCol = df1.shape
    print(f'There are {nRow} rows and {nCol} columns')

    print(df1.head(5))

    plotPerColumnDistribution(df1, 10, 5)

    # ================ #

    nRowsRead = 1000  # specify 'None' if want to read whole file
    # coronavirus_reddit_posts.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
    df2 = pd.read_csv('data/covid_reddit_posts/coronavirus_reddit_posts.csv', delimiter=',', nrows=nRowsRead)
    df2.dataframeName = 'coronavirus_reddit_posts.csv'
    nRow, nCol = df2.shape
    print(f'There are {nRow} rows and {nCol} columns')

    print(df2.head(5))

    plotPerColumnDistribution(df2, 10, 5)

    # ================ #

    plotCorrelationMatrix(df2, 8)

    # ================ #

    plotScatterMatrix(df2, 6, 15)

    # ================ #

    nRowsRead = 1000  # specify 'None' if want to read whole file
    # coronavirus_reddit_raw_comments.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
    df3 = pd.read_csv('data/covid_reddit_posts/coronavirus_reddit_raw_comments.csv', delimiter=',', nrows=nRowsRead)
    df3.dataframeName = 'coronavirus_reddit_raw_comments.csv'
    nRow, nCol = df3.shape
    print(f'There are {nRow} rows and {nCol} columns')

    print(df3.head(5))

    plotPerColumnDistribution(df3, 10, 5)
    # ================ #
