import pandas as pd
import numpy as np
import datetime

def make_integer_time_index(time_exch, config=None):
    '''
    Creates integer time index from 'time_exchange' column returned in format of Coin API

    This function is designed to be called with pd.Series.apply(lambda x: func(x, config=config)), x is the value of 'time_exch' passed on by apply function,
    while what kind of index will be created should be specified by user via 'config' param.

        args:

            * time_exch: this is passed by .apply function from pandas
            * config (str): user specifies which time index to create  as '[agg_type]_[agg_freq]':
                - agg_type:
                    = cycle:  only return the cycle. ( example if freq is day, then return day of the month example: 1, 2, 24, 30)
                    = bin: concatenate starting from largest freq(year) until specified freq (allows easy binning with this freq, hence the name 'bin')
    '''

    agg_type, agg_freq = config.split('_')

    t = str(time_exch).split('T')

    year, month, day = t[0].split('-')
    hour, minute, second = t[1].split('.')[0].split(':')
    ymd_hms = {'year': year, 'month': month, 'day': day, 'hour': hour, 'minute': minute, 'second': second}

    if agg_type == 'cycle':
        return (int(ymd_hms[agg_freq]))

    elif agg_type == 'bin':
        bin_time = ''
        for freq, freq_value in ymd_hms.items():
            bin_time += freq_value
            if freq == agg_freq:
                break
        return (int(bin_time))

def make_all_time_indexes(df):
    time_agg_inds = []

    for agg_type in ['cycle', 'bin']:
        for agg_freq in ['year', 'month', 'day', 'hour', 'minute', 'second']:
            print(agg_type, agg_freq)
            if agg_type == 'cycle':
                config = str(agg_type + '_' + agg_freq)
                df[agg_freq] = df['time_exchange'].apply(lambda x: make_integer_time_index(x, config=config))
                time_agg_inds.append(agg_freq)
            else:
                config = str('bin_' + agg_freq)
                df['time_' + agg_freq] = df['time_exchange'].apply(lambda x: make_integer_time_index(x, config=config))
                time_agg_inds.append('time_' + agg_freq)
    df['date'] = pd.to_datetime(df['time_exchange']).dt.date
    return df
