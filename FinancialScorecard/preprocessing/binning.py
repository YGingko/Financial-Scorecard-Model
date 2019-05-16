# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Chocolate on 2019/5/13
"""

import numpy as np
import pandas as pd


class ChiMerge(object):

    def __init__(self, x, y):
        self.y_name = y.columns[0]
        self.data = pd.concat([x, y], axis=1)

    @staticmethod
    def calc_chi2(df, total_col, bad_col, bad_rate):
        tmp = df.copy()
        tmp['expected'] = tmp[total_col].map(lambda x: bad_rate * x)
        return sum(pow(tmp['expected'] - tmp[bad_col], 2) / tmp['expected'])

    def bin_cutoff(self, col, confidence=3.841, max_bins=5):
        """
        Split the continuous variable using Chi-square value by specifying the max number of intervals or the minimum confidence

        Parameters:
        -----------
        col: splitted column

        confidence: the specified chi-square threshold, by default the degree of freedom is 1 and using confidence level as 0.95

        max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work
        -----------

        return: the combined bins
        """

        # Calculate the total Bad_Rate , the total sample number and bad number in each group
        bad_rate = self.data[self.y_name].mean()
        total_cnt = pd.DataFrame({'total_cnt': self.data.groupby(col)[self.y_name].count()})
        bad_cnt = pd.DataFrame({'bad_cnt': self.data.groupby(col)[self.y_name].sum()})
        regroup = total_cnt.merge(bad_cnt, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)

        # Initialize the bins
        bins = [[i] for i in sorted(set(self.data[col].dropna().unique()))]
        bin_cnt = len(bins)
        while bin_cnt > 1:
            chi2_values = []
            for b in bins:
                tmp = regroup.loc[regroup[col].isin(b)]
                chi2_values.append(self.calc_chi2(tmp, 'total_cnt', 'bad_cnt', bad_rate))

            # Determine when to stop interval merging
            if pd.isna(confidence):
                if pd.isna(max_bins):
                    confidence = 3.841
                    max_bins = 5
                else:
                    confidence = -np.inf
            else:
                if pd.isna(max_bins):
                    max_bins = np.inf

            if all([min(chi2_values) >= confidence, bin_cnt <= max_bins]):
                break

            # Merge interval
            min_position = chi2_values.index(min(chi2_values))
            if min_position == 0:
                com_position = 1
            elif min_position == bin_cnt - 1:
                com_position = min_position - 1
            elif chi2_values[min_position - 1] <= chi2_values[min_position + 1]:
                com_position = min_position - 1
            else:
                com_position = min_position + 1

            bins[min_position] = bins[min_position] + bins[com_position]
            bins.remove(bins[com_position])
            bin_cnt = len(bins)

        # Extract the cutoff of each box
        bins = [sorted(b) for b in bins]
        cutoff = [b[-1] for b in bins[:-1]]
        return cutoff

    @staticmethod
    def calc_woe_iv(df, col, target):
        """
        Calculate WOE and IV

        Parameters:
        -----------
        col: column after splitted
        -----------

        return: WOE, total IV, IV in each bin
        """

        # Calculate total number, good number, bad number in total sample
        bad = df[target].sum()
        good = df.shape[0] - bad

        # Calculate the total sample number and bad number in each group
        total_cnt = pd.DataFrame({'total_cnt': df.groupby(col)[target].count()})
        bad_cnt = pd.DataFrame({'bad_cnt': df.groupby(col)[target].sum()})
        regroup = total_cnt.merge(bad_cnt, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)

        regroup['good_cnt'] = regroup['total_cnt'] - regroup['bad_cnt']
        regroup['bad_pcnt'] = regroup['bad'] / bad
        regroup['good_pcnt'] = regroup['good_cnt'] / good
        regroup['WOE'] = np.log(regroup['bad_pcnt'] / regroup['good_pcnt'])
        regroup['IV'] = (regroup['bad_pcnt'] - regroup['good_pcnt']) * regroup['WOE']

        return regroup[['WOE', 'IV']]

    @staticmethod
    def badrate_monotone(df, col, target):
        """
        Check the monotone

        Parameters:
        -----------
        col: column, after splitted
        -----------

        return: bool, if the bad rate is monotone
        """
        data = df[df[col].notnull() & (df[col] != 'NA')]

        total_cnt = pd.DataFrame({'total_cnt': data.groupby(col)[target].count()})
        bad_cnt = pd.DataFrame({'bad_cnt': data.groupby(col)[target].sum()})
        regroup = total_cnt.merge(bad_cnt, left_index=True, right_index=True, how='left')
        regroup.sort_index(inplace=True)
        regroup.reset_index(level=0, inplace=True)

        regroup['bad_rate'] = regroup['bad_cnt'] / regroup['total_cnt']
        badrate_monotone = regroup['bad_rate'].diff().dropna().astype(bool)
        monotone = len(set(badrate_monotone))
        return monotone == 1

    def maximum_bin_pcnt(self, col, thd=0.9):
        """
        Determine whether the proportion of samples in the largest bin is greater than thd

        Parameters:
        -----------
        thd: the proportion of samples
        -----------

        return: bool, if the proportion is greater than thd
        """

        bin_cnt = self.data.groupby(col)[col].count()
        bin_pcnt = bin_cnt / self.data.shape[0]
        return bin_pcnt.max() >= thd

    def badrate_encoding(self, col):
        """
        For category data, use bad_rate to replace the original value, convert into a continuous variable and then band.
        There is no need to bin if the variable are fewer categories in principle.

        Parameters:
        -----------
        col: category column
        -----------

        return: replace by bad_rate
        """

        total_cnt = pd.DataFrame({'total_cnt': self.data.groupby(col)[self.y_name].count()})
        bad_cnt = pd.DataFrame({'bad_cnt': self.data.groupby(col)[self.y_name].sum()})
        regroup = total_cnt.merge(bad_cnt, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)

        regroup['bad_rate'] = regroup['bad_cnt'] / regroup['total_cnt']
        return self.data.merge(regroup[[col, 'bad_rate']], on=col, how='left')['bad_rate']


class WOE_IV:

    def __init__(self, df, key, y_name):
        self.set_data(df, key, y_name)
        self.gen_var_info()

    def set_data(self, df, key, y_name):
        self.data = df
        if key is not None:
            self.data = df.set_index(keys=key)
        self.y_name = y_name
        self.target = self.data[[self.target]]
        self.data.drop(self.y_name, axis=1, inplace=True)

    def gen_var_info(self):
        self.var_info = pd.DataFrame(self.data.nunique(), columns=['num_of_unique'])

    def cal(self, bin_cnt=10):
        self.gen_table(bin_cnt=10)



    def gen_table(self, bin_cnt):
        chi2 = ChiMerge(self.data, self.target)
        for col in self.data.columns:
            # binning
            cutoff = chi2.bin_cutoff(col, confidence=3.841, max_bins=bin_cnt)
            tmp = pd.cut(self.data[col], bins=[-np.inf] + cutoff + [np.inf]).cat.add_categories('NA').fillna('NA')
            tmp.value_counts().sort_index()


    def to_band(self, cols, max_bins):
        self.tables = {}

        chimerge = ChiMerge(self.data, y_name=self.target)
        for index in range(len(cols)):
            col = cols[index]
            max_bin = max_bins[index]

            # binning
            cutoff = chimerge.bin_cutoff(col, confidence=3.841, max_bins=max_bin)


            self.band_df[col] = pd.cut(self.data[col], bins=[-np.inf] + cutoff + [np.inf]).cat.add_categories('NA').fillna('NA')
            if chimerge.badrate_monotone(self.band_df, col, self.target):
                self.band_monotone_df[col] = self.band_monotone_df[col]
            else:
                print('column: ', col, 'band error, reason is not monotone')

            woe_iv = chimerge.calc_woe_iv(self.band_df, col, self.target)
            self.woe_df[col] = self.band_df[col].map(woe_iv['WOE'].to_dict())



