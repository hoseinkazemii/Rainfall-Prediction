import pandas as pd

def deal_with_outliers(df, **params):
	verbose = params.get("verbose")
	verbose_outliers = params.get("verbose_outliers")

	if verbose:
		print("dealing with outliers...")

	df['WindSpeed9am'] = pd.to_numeric(df['WindSpeed9am'])
	df['WindSpeed3pm'] = pd.to_numeric(df['WindSpeed3pm'])

	IQR_WindSpeed9am = float(df.WindSpeed9am.quantile(0.75)) - float(df.WindSpeed9am.quantile(0.25))
	IQR_WindSpeed3pm = float(df.WindSpeed3pm.quantile(0.75)) - float(df.WindSpeed3pm.quantile(0.25))

	rain_outlier = df.Rainfall.quantile(0.85)

	I1ma = (df.WindSpeed9am.quantile(0.75) + 1.5*IQR_WindSpeed9am)
	I1mi = (df.WindSpeed9am.quantile(0.25) - 1.5*IQR_WindSpeed9am)
	df_outlier_WindSpeed9am = [i for i in df['WindSpeed9am'] if i > I1ma or i < I1mi]
	
	if verbose_outliers:
		print('the number of "WindSpeed9am" outliers is:',len(df_outlier_WindSpeed9am))

	I2ma = (df.WindSpeed3pm.quantile(0.75) + 1.5*IQR_WindSpeed3pm)
	I2mi = (df.WindSpeed3pm.quantile(0.25) - 1.5*IQR_WindSpeed3pm)
	df_outlier_WindSpeed3pm = [i for i in df['WindSpeed3pm'] if i > I2ma or i < I2mi]

	if verbose_outliers:
		print('the number of "WindSpeed3pm" outliers is:',len(df_outlier_WindSpeed3pm))

	df_outlier_Rainfall = [i for i in df['Rainfall'] if i > rain_outlier]

	if verbose_outliers:
		print('the number of "Rainfall" outliers is:',len(df_outlier_Rainfall))

	df = df[(df['WindSpeed9am']>I1mi) & (df['WindSpeed9am']<I1ma)]
	df = df[(df['WindSpeed3pm']>I2mi) & (df['WindSpeed3pm']<I2ma)]
	df = df[(df['Rainfall']<rain_outlier)]

	return df
