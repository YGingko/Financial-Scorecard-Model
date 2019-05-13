# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Chocolate on 2019/5/13
"""


class FeatureSelection(object):

    def __init__(self, data, key=None, y_name=None):
        self.set_data(data, key, y_name)

    def set_data(self, data, key=None, y_name=None):
        self.y_name = y_name
        if key is None:
            self.data = data
        else:
            self.data = data.set_index(keys=key)

    def MissingThreshold(self, thd=0.8):

        df = self.data.copy()
        if self.y_name is not None:
            df.drop(self.y_name, axis=1, inplace=True)

        missing_rate = df.isnull().sum() / df.shape[0]

        print('The features dropped by missing:', missing_rate[missing_rate >= thd].index.tolist())

        col = missing_rate[missing_rate < thd].index.tolist()
        if self.y_name is not None:
            self.data = self.data[df[:, col].columns.tolist() + [self.y_name]]
        else:
            self.data = self.data[df[:, col].columns.tolist()]
        print('The Number of Features Selected By Missing:', len(col))
        return self.data

    def VarThreshold(self, thd=0):
        """
        Feature selector that removes all low-variance features.

        Parameters:
        -----------
        thd: numeric, threshold of the variance.
        -----------
        """

        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(thd)

        df = self.data
        if self.y_name is not None:
            df.drop(self.y_name, axis=1, inplace=True)

        selector.fit_transform(df)

        # Print names of low variance features
        col = [i for i, value in enumerate((selector.variances_ <= thd).tolist()) if value]
        print('The features dropped:', df[:, col].columns.tolist())

        # Keep high variance features
        col = [i for i, value in enumerate((selector.variances_ > thd).tolist()) if value]

        if self.y_name is not None:
            self.data = self.data[df[:, col].columns.tolist() + [self.y_name]]
        else:
            self.data = self.data[df[:, col].columns.tolist()]
        print('The Number of Features Selected By Variance:', len(col))

        return self.data
