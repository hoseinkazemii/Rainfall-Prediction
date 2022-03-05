
def describe_numerical_cols(df, **params):
	numerical_cols = [var for var in df.columns if df[var].dtype!='O']

	print(df[numerical_cols].describe(include = 'all'))