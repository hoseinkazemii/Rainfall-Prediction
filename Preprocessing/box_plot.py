import matplotlib.pyplot as plt


def box_plot(df, **params):
	box_plot_cols = params.get("box_plot_cols")

	plt.rcParams['figure.figsize'] = 10, 10
	df_outliers = df[box_plot_cols]
	df_outliers.boxplot(return_type = 'dict')
	
	plt.show()