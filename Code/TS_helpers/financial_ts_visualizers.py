import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def summarise_fin_ts(df, forecasted_col, volume_col, show = 0.3):


    with plt.style.context('bmh'):
        last_ind = int(show * df.shape[0])
        t = df[:last_ind].index

        fig = plt.figure(figsize=(15, 10))
        layout = (4, 2)
        plot1 = plt.subplot2grid(layout, (0, 0), colspan=2)
        plot2 = plt.subplot2grid(layout, (1, 0), colspan=2)
        plot3 = plt.subplot2grid(layout, (2, 0), colspan=2)
        plot4 = plt.subplot2grid(layout, (3, 0), colspan=2)

        plot1.plot(t, (daily.weighted_price / daily.weighted_price.max())[:last_ind], color='blue')
        #     plot1.plot(t, exp_smooth(daily.weighted_price, alpha=0.1)[:last_ind], color='lightblue')

        plot2.plot(t, (daily_log_price / daily_log_price.max())[:last_ind])
        plot2.bar(t, (daily_log_rets / daily_log_rets.max())[:last_ind], color=daily_trend[:last_ind])

        plot3.fill_between(t, 0,
                           exp_smooth(daily.volume_cur.apply(np.log) / daily.volume_cur.apply(np.log).max())[:last_ind],
                           color='lightblue')
        plot3.fill_between(t, 0, (daily.volume_cur / daily.volume_cur.max())[:last_ind], color='blue')

        #     plot2.fill_between(s1[:last_ind].index, 0, (daily.volume_btc / daily.non_na_minute_counts)[:last_ind],color='blue')

        plot4.plot(t, exp_smooth(daily.non_na_minute_counts / daily.non_na_minute_counts.max(), 0.1)[:last_ind],
                   color='darkgrey')
        #     exp_smooth((daily.volume_cur/daily.non_na_minute_counts)/(daily.volume_cur/daily.non_na_minute_counts).max(), 0.05)[:last_ind].plot(ax=plot4,color='blue')

        for plot in [plot1, plot2, plot3, plot4]:
            plot.set_xlim(daily_log_rets.index[0], daily_log_rets.index[last_ind - 1])
        plt.tight_layout()
