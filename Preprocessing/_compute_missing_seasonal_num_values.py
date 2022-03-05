import numpy as np

def _compute_missing_seasonal_num_values(df, col):
    #separate to seasons
    temp9am_mean = df['Temp9am'].mean()
    temp3pm_mean = df['Temp3pm'].mean()
    min_temp_mean = df['MinTemp'].mean()
    max_temp_mean = df['MaxTemp'].mean()
    
    defaults = {'Temp9am': temp9am_mean, 'Temp3pm': temp3pm_mean, 'MinTemp': min_temp_mean, 'MaxTemp': max_temp_mean}
    
    spring = df[(df['Month']) >= 9 & (df['Month'] < 12)]
    sp_mean = np.mean(spring.loc[:,col])
    if (np.isnan(sp_mean) == True):
        sp_mean = defaults[col]
    
    summer = df[((df['Month'] >= 1) | (df['Month'] < 3)) & (df['Month'] == 12)]
    sm_mean = np.mean(summer.loc[:,col])
    if (np.isnan(sm_mean) == True):
        sm_mean = defaults[col]
    
    fall = df[(df['Month'] >= 3) & (df['Month'] < 6)]
    fa_mean = np.mean(fall.loc[:,col])
    if (np.isnan(fa_mean) == True):
        fa_mean = defaults[col]

    winter = df[(df['Month'] >= 6) & (df['Month'] < 9)]
    wt_mean = np.mean(winter.loc[:,col])
    if (np.isnan(wt_mean) == True):
        wt_mean = defaults[col]

    return sp_mean, sm_mean, fa_mean, wt_mean