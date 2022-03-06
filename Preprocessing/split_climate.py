
def split_climate(df, **params):
	Koppen_climate = params.get("Koppen_climate")
	verbose = params.get("verbose")

	if verbose:
		print(f"splitting data to two dataframes:\n1.df with {Koppen_climate} Koppen climate class and 2.df with all climate classes excluding {Koppen_climate}")

	df_climate = df[df[f'Koppen_{Koppen_climate}'] == 1]
	df_all = df[df[f'Koppen_{Koppen_climate}'] != 1]

	return df_climate, df_all