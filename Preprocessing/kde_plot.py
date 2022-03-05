import matplotlib.pyplot as plt
import seaborn as sns

def kde_plot(df, **params):
	plt.figure(figsize = (10,10))
	sns.kdeplot(data = df['Rainfall'], shade = True)

	plt.show()