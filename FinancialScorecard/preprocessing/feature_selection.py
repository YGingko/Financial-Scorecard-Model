# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Chocolate on 2019/5/13
"""


def missing_threshold(data, y_name=None, thd=0.8):
    df = data.copy()
    if y_name is not None:
        df.drop(y_name, axis=1, inplace=True)

    missing_rate = df.isnull().sum() / df.shape[0]

    print('The features dropped by missing:', missing_rate[missing_rate >= thd].index.tolist())

    col_names = missing_rate[missing_rate < thd].index.tolist()
    print('The Number of Features Selected By Missing:', len(col_names))
    if y_name is not None:
        col_names = col_names + [y_name]
    return data[col_names]


def var_threshold(data, y_name=None, thd=0):
    """
    Feature selector that removes all low-variance features.

    Parameters:
    -----------
    thd: numeric, threshold of the variance.
    -----------
    """

    from sklearn.feature_selection import VarianceThreshold

    df = data.copy()
    if y_name is not None:
        df.drop(y_name, axis=1, inplace=True)
    selector = VarianceThreshold(thd)
    selector.fit_transform(df)

    # Print names of low variance features
    col = [i for i, value in enumerate((selector.variances_ <= thd).tolist()) if value]
    print('The features dropped:', df.iloc[:, col].columns.tolist())

    # Keep high variance features
    col = [i for i, value in enumerate((selector.variances_ > thd).tolist()) if value]
    print('The Number of Features Selected By Variance:', len(col))

    col_names = df.iloc[:, col].columns.tolist()
    if y_name is not None:
        col_names = col_names + [y_name]
    return data[col_names]

