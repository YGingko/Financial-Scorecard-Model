# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by Chocolate on 2019/5/13
"""

import os
import numpy as np
import pandas as pd


class ChiMerge(object):

    def __init__(self, x, y):
        self.y_name = y.columns[0]
        self.data = pd.concat([x, y], axis=1)

    def calc_chi2(self, df, total_col, bad_col, bad_rate):
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

    def __init__(self, df, y_name, key=None):
        self.data = df.copy()
        if key is not None:
            self.data = df.set_index(keys=key)
        self.y_name = y_name
        self.target = self.data[[self.y_name]]
        self.data.drop(self.y_name, axis=1, inplace=True)
        self.tables = {}
        self.var_info = pd.DataFrame()
        self.cut_off = {}

    def cal(self, bin_cnt=10):
        self.__gen_table(bin_cnt=10)
        self.__cal_table()
        self.__gen_var_info()
        self.__woe_replacement()

    def __gen_table(self, bin_cnt):
        chi2 = ChiMerge(self.data, self.target)
        for col in self.data.columns:
            # binning
            cutoff = chi2.bin_cutoff(col, confidence=3.841, max_bins=bin_cnt)
            self.cut_off[col] = cutoff

            tmp = self.data[[col]].copy()
            tmp['bin_cut'] = pd.cut(tmp[col], bins=[-np.inf] + cutoff + [np.inf]).cat.add_categories('NA').fillna('NA')
            tmp = pd.concat([tmp, self.target], axis=1)

            num_table = tmp.pivot_table(index='bin_cut', values=col, aggfunc=[np.median, len])
            num_table.columns = num_table.columns.droplevel(1)
            num_table.rename(columns={'len': 'size', 'median': 'bin_label'}, inplace=True)

            flag_table = tmp.pivot_table(index='bin_cut', values=col, columns=self.y_name, aggfunc=len)
            table = pd.concat([num_table, flag_table], axis=1)
            table.index.name = 'bin_cut'
            table.reset_index(level=0, inplace=True)
            self.tables[col] = table

    def __cal_table(self):
        for col, table in self.tables.items():
            self.tables[col]['bad_rate'] = (table[1] / table['size']).fillna(0)
            self.tables[col]['bad_pcnt'] = table[1] / table[1].sum()
            self.tables[col]['good_pcnt'] = table[0] / table[0].sum()
            self.tables[col]['sample_pcnt'] = table['size'] / table['size'].sum()
            self.tables[col]['WOE'] = np.log(self.tables[col]['bad_pcnt'] / self.tables[col]['good_pcnt'])
            self.tables[col]['IV'] = (self.tables[col]['bad_pcnt'] - self.tables[col]['good_pcnt']) * self.tables[col]['WOE']

    def __gen_var_info(self):
        self.var_info = pd.DataFrame(self.data.nunique(), columns=['num_of_unique'])

        missing_rate = {}
        num_of_bins = {}
        iv_value = {}
        is_monotone = {}
        for col, table in self.tables.items():
            missing_rate[col] = table['size'][0] / table['size'].sum() if 'NA' in table['bin_cut'] else 0
            num_of_bins[col] = table.shape[0]
            iv_value[col] = table['IV'].sum()
            is_monotone[col] = int(self.__badrate_monotone(table))
        self.var_info['iv_value'] = pd.Series(iv_value)
        self.var_info['iv_rank'] = self.var_info['iv_value'].rank(method='min')
        self.var_info['missing_rate(%)'] = np.round(pd.Series(missing_rate), 1)
        self.var_info['num_of_bins'] = pd.Series(num_of_bins)
        self.var_info['is_monotone'] = pd.Series(is_monotone)
        self.var_info.index.name = 'var_name'

    def __woe_replacement(self):
        for col, table in self.tables.items():
            self.data[col] = self.data[col].map(table['WOE'].to_dict())

    def __badrate_monotone(self, table):
        table = table[table['bin_cut'] != 'NA']
        monotone = table['bad_rate'].diff().dropna().astype(bool)
        return len(set(monotone)) == 1

    def save_info(self, path='WOE_IV_Results'):
        os.makedirs(path, exist_ok=True)
        self.var_info.sort_values(by='iv_rank').to_excel(path + '\\var_info_table.xlsx')
        with open(path + '\\var_info.txt', 'w') as file:
            for col, table in self.tables.items():
                file.write(col + '\n')
                table.to_string(file)
                file.write('\n' * 2)
        print('Done.')

    def iv_threshold(self, thd=0.01):
        temp = self.var_info[self.var_info['iv_value'] >= thd].index
        return temp.values

    def get_data(self, monotone=False, max_bin_pcnt=None):
        tmp = self.data.copy()

        if monotone:
            cols = self.var_info[self.var_info['is_monotone']].index.values
            tmp = tmp[cols]

        if max_bin_pcnt is not None:
            cols = []
            for col, table in self.tables.items():
                if table['sample_pcnt'].max() < max_bin_pcnt:
                    cols.append(col)
            tmp = tmp[cols]
        return pd.concat([tmp, self.target], axis=1)

    def transform_data(self, test_data):
        test_data_woe = test_data.copy()
        for col, cutoff in self.cut_off.items():
            test_data_woe[col] = pd.cut(test_data_woe[col], bins=[-np.inf] + cutoff + [np.inf]).cat.add_categories('NA').fillna('NA').map(self.tables[col]['WOE'].to_dict())
        return test_data_woe

