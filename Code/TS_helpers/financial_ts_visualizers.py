import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.api import SimpleExpSmoothing


def summarise_BTC1min(df, col_val,
                      col_volume,
                      time_first,
                      time_last,
                      cols_smooth=[],
                      smoothing=0.1):

    if time_first is not None:
        mask = (df.index >= time_first)
    if time_last is not None:
        mask = (df.index <= time_last)

    # Time ticks
    t = df[mask].index

    # Price normalized
    val_max_normalized = df[mask][col_val]/df[mask][col_val].max()
    # Log Price
    val_log = df[mask][col_val].apply(np.log)
    if col_val in  cols_smooth:
        val_max_normalized = SimpleExpSmoothing(val_max_normalized).fit(smoothing_level=smoothing).fittedvalues
        val_log = SimpleExpSmoothing(val_log).fit(smoothing_level=smoothing).fittedvalues
    # Log returns
    log_returns = val_log - val_log.shift()
    # Log returns trend
    log_ret_trend = (log_returns < 0).apply(lambda x: 'red' if x else 'green')

    # Volume

    volume = df[mask][col_volume]
    if col_volume in cols_smooth:
        volume = SimpleExpSmoothing(volume).fit(smoothing_level=smoothing).fittedvalues
    # Log volume
    log_volume = volume.apply(np.log)

    # None na minutes
    non_na_mins = df[mask]['count_non_na_mins']
    if 'count_non_na_mins' in cols_smooth:
        non_na_mins = SimpleExpSmoothing(non_na_mins).fit(smoothing_level=smoothing).fittedvalues

    with plt.style.context('bmh'):
        
        plt.figure(figsize=(15, 10))
        layout = (4, 2)
        value_plot = plt.subplot2grid(layout, (0, 0), colspan=2)
        log_returns_plot = plt.subplot2grid(layout, (1, 0), colspan=2)
        volume_plot = plt.subplot2grid(layout, (2, 0), colspan=2)
        non_na_mins_plot = plt.subplot2grid(layout, (3, 0), colspan=2)

        # Value plot itself
        value_plot.plot(t, val_max_normalized , color='blue')

        # Log price (max normalized)
        value_plot.plot(t, val_log/val_log.max())
        
        # Log returns with(trend)
        log_returns_plot.bar(t, log_returns, color=log_ret_trend)

        # Volume plots
        ## Log Volume (max normalized)
        volume_plot.fill_between(t, 0, (log_volume / log_volume.max()), color='lightblue')
        ## Volume plot (max normalized)
        volume_plot.fill_between(t, 0, volume / volume.max(), color='blue')

        # Non Na minutes (max normalized)
        non_na_mins_plot.plot(t, non_na_mins/non_na_mins.max(), color='darkgrey')


        for plot in [value_plot, log_returns_plot, volume_plot, non_na_mins_plot]:
            plot.set_xlim(t[0], t[-1])
        plt.tight_layout()
