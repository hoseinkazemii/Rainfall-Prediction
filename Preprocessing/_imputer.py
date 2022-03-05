import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer


def _imputer(df, col, **params):
	imputer = params.get("imputer")

	if imputer == "KNN":
		knn_imputer = KNNImputer(n_neighbors = 5, weights = "uniform")
		df[col] = knn_imputer.fit_transform(df[[col]])
	
	elif imputer == "Simple":
		simple_imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
		df[col] = simple_imputer.fit_transform(df[[col]])

	elif imputer == "Iter_BR":
		iter_imputer = IterativeImputer()
		df[col] = iter_imputer.fit_transform(df[[col]])

	elif imputer == "Iter_DT":
		iter_imputer = IterativeImputer(estimator = DecisionTreeRegressor(max_features = 'sqrt', random_state = 42))
		df[col] = iter_imputer.fit_transform(df[[col]])

	elif imputer == "Iter_ET":
		iter_imputer = IterativeImputer(estimator = ExtraTreesRegressor(n_estimators = 10, random_state = 42))
		df[col] = iter_imputer.fit_transform(df[[col]])

	else:
		print('select one imputer from list:\
			["KNN","Simple","Iter_BR","Iter_DT","Iter_ET"]')

	return df