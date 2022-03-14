
def split_climate(df, **params):
	Koppen_climate = params.get("Koppen_climate")
	verbose = params.get("verbose")
	CB = params.get("CB")

	if verbose:
		print(f"splitting data to two dataframes:\n1.df with {Koppen_climate} Koppen climate class and 2.df with all climate classes excluding {Koppen_climate}")

	if not CB:
		df_climate = df[df[f'Koppen_{Koppen_climate}'] == 1]
		df_all = df[df[f'Koppen_{Koppen_climate}'] != 1]
	else:
		df_climate = df[df['Koppen'] == Koppen_climate]
		df_all = df[df['Koppen'] != Koppen_climate]		

	return df_climate, df_all