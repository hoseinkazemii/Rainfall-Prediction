from ._compute_missing_seasonal_num_values import _compute_missing_seasonal_num_values

import pandas as pd

def _fill_missing_seasonal_num_values(df, location, col):
    dfs = []
        
    sp, sm, fa, wt = _compute_missing_seasonal_num_values(df[df['Location'] == location], col)
    df = df[df['Location'] == location]

    sp_df = df[(df['Month'] >= 9) & (df['Month'] < 12)]
    sp_df[col].fillna(sp, inplace=True)
    
    sm_df = df[((df['Month'] >= 1) & (df['Month'] < 3)) | (df['Month'] == 12)]
    sm_df[col].fillna(sm, inplace=True)
    
    fa_df = df[(df['Month'] >= 3) & (df['Month'] < 6)]
    fa_df[col].fillna(fa, inplace=True)
    
    wt_df = df[(df['Month'] >= 6) & (df['Month'] < 9)]
    wt_df[col].fillna(wt, inplace=True)

    dfs.append(sp_df)
    dfs.append(sm_df)
    dfs.append(fa_df)
    dfs.append(wt_df)

    df = pd.concat(dfs)
        
    return df