import numpy as np
import pandas as pd


def exp_smooth(seq, alpha=0.1):
    exp_smooth = [np.nan]

    first_val_appended = False
    for t in range(0, len(seq.index) - 1):
        if np.isnan(seq[t]):
            exp_smooth.append(np.nan)
            continue
        elif not first_val_appended:
            exp_smooth.append(seq[t])
            first_val_appended = True
            continue

        exp_smooth.append((1 - alpha) * exp_smooth[-1] + alpha * seq[t])

    exp_smooth = pd.Series(index=seq.index, data=exp_smooth)
    return exp_smooth


def print_rmse(df, col_pred, col_pred_real, masks, data_divs = ['train', 'valid']):
    for data_div in data_divs:
        mask = masks[data_div]
        print(data_div + " set error \n"+str(np.sqrt(np.mean((df[col_pred]-df[mask][col_pred_real])**2)).round(4)))


