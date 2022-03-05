
def _cat_nulls(df, **params):
	categorical = [col for col in df.columns if df[col].dtypes == 'O']

	for col in categorical:
	    col_mode = df[col].mode()[0]
	    df[col].fillna(col_mode, inplace = True)

	return df