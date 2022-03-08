from sklearn.utils import shuffle

def shuffle_df(df, **params):
	df = shuffle(df)

	return df