from DataLoader import *
from Preprocessing import *
from MLModels import *

def run():
	settings = {
	"data_path" : "./Data/Weather.csv",
	"box_plot_cols" : ['Rainfall','WindSpeed9am','WindSpeed3pm'],
	"verbose" : True,
	"verbose_outliers" : True,
	"imputer" : "Iter_BR",
	"K_SMOTE" : 5,
	"model" : "CatBoost",





	}


	#Step1-Preprocessing:

	df = load_data(**settings)
	df = make_datetime_cols(df, **settings)
	# describe_numerical_cols(df, **settings)
	# box_plot(df, **settings)
	# kde_plot(df, **settings)
	df = deal_with_outliers(df, **settings)
	df = deal_with_nulls(df, **settings)
	df = cat_dummies(df, **settings)
	X_train, X_test, y_train, y_test = split_data(df, **settings)
	X_train, X_test = scaler(X_train, X_test, **settings)
	X_train, y_train = oversample(X_train, y_train, **settings)

	#Step2-Training:
	# 2-1: CatBoost
	cb_settings = {'iterations' : 2,
					'learning_rate' : 0.1,
					'depth' : 9,
					'l2_leaf_reg' : 0.001,
					'loss_function' : 'Logloss',
					# 'loss_function' : 'RMSE',
					'allow_writing_files' : False,
					# 'eval_metric' : "RMSE",
					'eval_metric' : "Accuracy",
					'task_type' : 'CPU',
					'verbose' : 400,
					'boosting_type' : 'Ordered',
					'thread_count' : -1,}

	myCatBoostModel = CatBoostModel(**{**cb_settings,
		                                          **settings})
	myCatBoostModel._construct_model()
	myCatBoostModel.run(X_train, X_test, y_train, y_test)

	# 2-2:RandomForest
	# rf_settings = {'n_estimators' : 100,
	# 				'max_depth' : 200,
	# 				'min_samples_split' : 2,
	# 				'min_samples_leaf' : 1,
	# 				'max_features' : 'auto',
	# 				'should_cross_val' : False,
	# 				'n_jobs' : -1,}

	# myRFModel = RFModel(**{**rf_settings,
	# 									**modelling_settings})
	# myRFModel._construct_model()
	# myRFModel.run()















if __name__ == "__main__":
	run()
