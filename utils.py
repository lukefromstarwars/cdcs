import time
from importlib import reload
from itertools import combinations
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors, cm, rcParams
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures


pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 180)
pd.set_option('max_colwidth', 180)
desired_width = 320
pd.set_option('display.width', desired_width)

# -- GRAPHICS
# ------------------------------------------------------------------------------------------

# matplotlib.style.use('fivethirtyeight')
# sns.set(style='ticks', color_codes=True)

sns.set(style="whitegrid")
sns.set_context(rc={'lines.linewidth': .4, 'grid.linewidth': .2})

cmap_excel_1 = colors.ListedColormap(['#4F81BD', '#C0504D', '#9BBB59', '#8064A2', '#4BACC6', '#F79646', '#2C4D75', '#772C2A', '#5F7530', '#276A7C'])
cmap_excel_2 = colors.ListedColormap(['#4F81BD', '#9BBB59', '#4BACC6', '#2C4D75', '#5F7530', '#276A7C', '#729ACA', '#AFC97A', '#6FBDD1', '#3A679C'])
cmap = cm.get_cmap('viridis')

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Calibri', 'Tahoma']

# -- ENVIRONMENT
# ------------------------------------------------------------------------------------------
graphs_folder = 'GRAPHS/'
graph_clusters_folder = 'GRAPH_CLUSTERS/'
excel_folder = 'EXCEL/'
pickles_folder = 'PICKLES/'
html_folder = 'HTML/'

# -- VARIABLES
# ------------------------------------------------------------------------------------------
# strings
lst_name = 'LST_NAME'
lst_value = 'LST_VALUE'

# project columns
lst_fr = 'LST_FR'
lst_nl = 'LST_NL'
lst_en = 'LST_EN'
var = 'VAR'
var_name = 'VAR_NAME'
var_code = 'VAR_CODE'
var_type = 'VAR_TYPE'
var_lst = 'VAR_LST'
var_desc = 'VAR_DESC'
var_used = 'VAR_USED'

# general columns
col_count = 'COUNT'
col_percent_na = 'PERCENT_NA'
col_percent_filled = 'PERCENT_W_VALUE'
col_total = 'TOTAL'
col_to_drop_from_var_list = [var_name, var_type, var_lst]

# general values
not_available = 'N/A'

the_years = [2011, 2012, 2013, 2014, 2015]


class BCOLORS:
	HEADER = '\033[95m'
	C_BLUE_NOBKG = '\033[94m'
	C_GREEN_NOBKG = '\033[92m'
	C_BLACK = '\033[30m'
	C_RED = '\033[31m'
	C_GREEN = '\033[32m'
	C_YELLOW = '\033[33m'
	C_BLUE = '\033[34m'
	C_MAGENTA = '\033[35m'
	C_CYAN = '\033[36m'
	C_WHITE = '\033[37m'
	C_RESET = '\033[39m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def rd_all():
	reload()


def get_methods_n_attributes(obj):
	print([method for method in dir(obj) if callable(getattr(obj, method))])


# region FILESAVE
# ------------------------------------------------------------------------------------------

def save_as_pickle(df, fn):
	file_path = pickles_folder + fn + '.pickle'
	print(file_path)
	pd.to_pickle(df, file_path)


def save_as_csv(df, fn):
	file_path = excel_folder + fn + '.csv'
	print(file_path)
	df.to_csv(file_path, decimal=',')


def save_as_xlsx(df, fn, nb_of_rows=0):
	if nb_of_rows > 0:
		df = df[:nb_of_rows]

	filepath = excel_folder + fn + '.xlsx'
	writer = pd.ExcelWriter(filepath, engine='xlsxwriter', date_format='dd/mmm/yyyy')

	# print df on sheet1
	df.to_excel(writer, 'Data')

	# print column names in alphabetical order on sheet 2
	df_cols = pd.DataFrame(df.columns)
	df_cols.to_excel(writer, 'Columns')

	print(filepath)
	writer.save()


def save_plot_as_png(plt_fig, fn, **kwargs):
	file_path = graphs_folder + fn + '.png'
	plt_fig.savefig(file_path, format='png', transparent=True, dpi=300, **kwargs)


def read_pickle(fn):
	return pd.read_pickle(pickles_folder + fn + '.pickle')


# endregion


# region STRINGS
# ------------------------------------------------------------------------------------------
def as_percent(v, precision='0.2'):
	"""Convert number to percentage string."""
	if isinstance(v, Number):
		return "{{:{}%}}".format(precision).format(v)
	else:
		raise TypeError("Numeric type required")


def as_no_decimal(v, precision='0.2'):
	"""Convert number to percentage string."""
	if isinstance(v, Number):
		return "{{:{}}}".format(precision).format(v)
	# return "{{:{}}}".format(precision).format(v)
	else:
		raise TypeError("Numeric type required")


def format_cols_as_pct(df):
	cols = get_numerical_cols(df)
	for col in cols:
		df[col] = df[col].apply(as_percent)

	return df


def color_negative_red(val):
	"""	Takes a scalar and returns a string with the css property `'color: red'` for negative strings, black otherwise."""
	color = 'red' if val < 0 else 'black'
	return 'color: %s' % color


def color_below_100_pct_red(val):
	"""	Takes a scalar and returns a string with the css property `'color: red'` for negative strings, black otherwise."""
	color = 'red' if val < 1 else 'black'
	return 'color: %s' % color


def left(s, pos):
	return s[:pos]


def right(s, pos):
	return s[-pos:]


def mid(s, offset, pos):
	return s[offset:offset + pos]


def is_number(n):
	return isinstance(n, (int, float, complex))


# endregion


# region COLUMNS STATS
# ------------------------------------------------------------------------------------------
def count_unique_categories_per_cols(df):
	"""
		Decide which categorical variables you want to use in model
		Count how many actual categories are represented in each of the dataframe columns
	"""

	for col_name in df.columns:
		if df[col_name].dtypes == 'object':
			unique_cat = len(df[col_name].unique())
			print("Feature '{col_name}' : {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))


def get_col_unique_counts(col, df):
	unique_categories = df[col].unique().tolist()
	if np.nan in unique_categories:
		unique_categories = [x for x in unique_categories if x is not np.NaN]
		unique_categories.append('N/A')
	# -- print(type(examples))

	unique_categories = sorted(unique_categories)
	print('COL:', col, ' Nb of CAT:', len(unique_categories), ' EX :', unique_categories[:100])


def get_cols_unique_counts(cols, df):
	cols.sort()
	# print(cols, df)
	for col in cols:
		get_col_unique_counts(col, df)


def get_unique_values(col, df):
	return sorted(df[col].unique().tolist())


def drop_columns(cols_to_drop, df):
	"""Drops columns from dataframe
	Args:
	    df (DataFrame): dataframe from which to drop columns
	    cols_to_drop(list): columns to be dropped
	Returns:
	    df: returns the modified dataframe
	"""
	cols = get_cols_alphabetically(df)

	for col in cols_to_drop:

		if col in cols:
			df = df.drop(col, axis=1)
			print("{}Dropped : {}{}".format(BCOLORS.WARNING, col, BCOLORS.ENDC))

	return df


def rename_column_to(df, old_name, new_name):
	df.rename(columns={old_name: new_name}, inplace=True)


def recode_col_to_num(df, col_to_encode, col):
	le = LabelEncoder()
	le.fit(df[col_to_encode].values)
	df[col] = le.transform(df[col_to_encode])


def print_full(df):
	pd.set_option('display.max_rows', len(df))
	pd.set_option('display.precision', 2)
	pd.set_option('expand_frame_repr', False)
	pd.set_option('display.max_columns', None)
	print(df)
	pd.reset_option('display.max_rows')
	pd.reset_option('expand_frame_repr')
	pd.reset_option('display.precision')
	pd.reset_option('display.max_columns')


def print_full_rows(df, nb_rows):
	print_full(df[:nb_rows])


def get_cols(df):
	return list(df.columns)


def get_col_count(col, nb_rows='All'):
	if nb_rows == 'All':
		this_count = col.value_counts().sort_values(ascending=False)
	else:
		this_count = col.value_counts().sort_values(ascending=False).head(nb_rows)
	print(this_count, len(this_count))


def get_nulls_count_per_cols(df):
	df_count = df.isnull().sum().sort_values(ascending=False)
	return df_count


def get_count_per_cols(df):
	df_count = df.count().sort_values(ascending=False)
	return df_count


def dummy_df(df, cols_to_dummy):
	"""Dummies all the categorical variables used for modeling from the column list"""
	nb_rows = len(df.columns)
	print('{} columns in DataFrame'.format(nb_rows))
	print('{} columns to dummy'.format(len(cols_to_dummy)))

	for col in cols_to_dummy:
		print('{}{} is dummied{}'.format(BCOLORS.WARNING, col, BCOLORS.ENDC))
		dummies = pd.get_dummies(df[col], prefix='DUM_' + col, dummy_na=False)
		df = df.drop(col, 1)
		df = pd.concat([df, dummies], axis=1)

	print('{}{} columns added in DataFrame{}'.format(BCOLORS.C_GREEN, len(df.columns) - nb_rows, BCOLORS.ENDC))

	return df


def add_power_col(df, cols, pwr):
	cols_num = get_numerical_cols(df)

	for col in cols:
		if col in cols_num:
			col_power = '{}_PWR_{}'.format(col, pwr)
			df[col_power] = df[col] ** pwr
			print('{}add col: {}{}'.format(BCOLORS.WARNING, col_power, BCOLORS.ENDC))

	return df


def add_log_col(df, cols):
	cols_num = get_numerical_cols(df)

	for col in cols:
		if col in cols_num:
			col_power = '{}_LOG'.format(col, )
			df[col_power] = np.log(df[col])
			print('{}add col: {}{}'.format(BCOLORS.WARNING, col_power, BCOLORS.ENDC))

			col_power = '{}_LOG2'.format(col)
			df[col_power] = np.log2(df[col])
			print('{}add col: {}{}'.format(BCOLORS.WARNING, col_power, BCOLORS.ENDC))

			col_power = '{}_LOG10'.format(col)
			df[col_power] = np.log10(df[col])
			print('{}add col: {}{}'.format(BCOLORS.WARNING, col_power, BCOLORS.ENDC))

	return df


def get_non_numerical_cols(df):
	"""Get list of non numerical columns in the dataframe"""
	non_numerical_cols = []

	for col_name in df.columns:
		if df[col_name].dtypes not in ['int32', 'int64', 'float32', 'float64']:
			non_numerical_cols.append(col_name)

	return non_numerical_cols


def get_numerical_cols(df):
	"""Get list of non numerical columns in the dataframe"""
	numerical_cols = []

	for col_name in df.columns:
		if df[col_name].dtypes in ['int32', 'int64', 'float32', 'float64']:
			numerical_cols.append(col_name)

	return numerical_cols


def get_stats(group):
	return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}


def get_unique_rows(cols, df, with_total=False):
	total = 'TOTAL'
	df = df[cols].groupby(cols).size().reset_index()
	df.columns = cols + [total]

	if with_total:
		df.loc[total] = df.sum()
		total_value = df[total][total]
		return df, total_value

	else:
		return df


def get_unique_rows_by_column(df):
	for col in sorted(df.columns):
		print(get_unique_rows([col], df))


# endregion


# region COLUMNS NAMING MANIPULATION
# ------------------------------------------------------------------------------------------
def get_cols_alphabetically(df):
	cols_txt = ['{}'.format(col) if (isinstance(col, (int, float, complex))) else col for col in df.columns]
	return sorted(cols_txt)


def get_cols_with(df, s, match_case=False):
	lst = get_cols_alphabetically(df)

	if not match_case:
		lst = [x.upper() for x in lst]
		s = s.upper()

	return [x for x in lst if s in x]


def col_replace_in_name(df, old, replacement):
	for col in df.columns:
		if old in col:
			new_col = col.replace(old, replacement)
			df.rename(columns={col: new_col}, inplace=True)
			print(col, new_col)


def col_prefix_to_suffix(df, prefix, suffix):
	for col in df.columns:
		# print(col, prefix, suffix)

		if left(col, len(prefix)) == prefix:
			new_col = right(col, len(col) - len(prefix))
			new_col += suffix
			rename_column_to(df, col, new_col)
			print(col, new_col)


def col_add_suffix(df, prefix, suffix):
	for col in df.columns:
		# print(col, prefix, suffix)

		if left(col, len(prefix)) == prefix:
			new_col = col + suffix
			rename_column_to(df, col, new_col)
			print(col, new_col)


def col_add_prefix(df, prefix, new_prefix):
	for col in df.columns:
		# print(col, prefix, suffix)

		if left(col, len(prefix)) == prefix:
			new_col = new_prefix + col
			rename_column_to(df, col, new_col)
			print(col, new_col)


def get_cols_with_prefix(df, prefix):
	cols = []
	for col in df.columns:
		if left(col, len(prefix)) == prefix:
			cols.append(col)
	return cols


def get_cols_with_suffix(df, suffix):
	cols = []
	for col in df.columns:
		if right(col, len(suffix)) == suffix:
			cols.append(col)
	return cols


def get_cols_with_prefix_suffix(df, prefix, suffix):
	cols = []
	for col in df.columns:
		if (left(col, len(prefix)) == prefix) & (right(col, len(suffix)) == suffix):
			cols.append(col)
	return cols


def set_cols_to_str(df):
	col_names = ['{}'.format(col) for col in df.columns]
	df.columns = col_names
	return df


# def remove_duplicate_cols(df, remove_pos='first'):
# 	cols_duplicates = list_duplicate_columns(df)
#
# 	print(cols_duplicates)
#
# 	for col in cols_duplicates:
# 		col_idxs = np.where(df.columns == col)
# 		print(col_idxs)
# 		if remove_pos == 'first':
# 			idx = np.array(col_idxs).min()
# 			print('{}Remove first column : {}{}'.format(BCOLORS.WARNING, col, BCOLORS.ENDC))
#
# 		else:
# 			idx = np.array(col_idxs).max()
# 			print('{}Remove last column : {}{}'.format(BCOLORS.WARNING, col, BCOLORS.ENDC))
#
# 		print(idx)
# 		df = df.drop(df.columns[idx], axis=1)
#
# 	return df


def list_duplicate_columns(df):
	cols_original = get_cols_alphabetically(df)
	cols_duplicates = list(set([x for x in cols_original if cols_original.count(x) > 1]))
	print('{}Duplicate columns : {}{}'.format(BCOLORS.WARNING, cols_duplicates, BCOLORS.ENDC))
	return cols_duplicates


def remove_col_with_prefix(df, prefix):
	cols = get_cols_with_prefix(df, prefix)

	print(prefix, cols)
	return drop_columns(cols, df)


def remove_col_with_suffix(df, suffix):
	cols = get_cols_with_suffix(df, suffix)

	print(suffix, cols)
	return drop_columns(cols, df)


def reorder_col_alphabetically(df):
	return df[sorted(df.columns)]


# endregion


# region COLUMNS MANIPULATION
# ------------------------------------------------------------------------------------------

def remove_values(lst, col, df):
	cnt = len(df)
	initial_cnt = len(df)

	for v in lst:
		df = df[df[col] != v]

		removed_cnt = cnt - len(df)
		cnt = len(df)
		print('{}Removed from \'{}\' {} values of type \'{}\' = {}{}'.format(BCOLORS.WARNING,
																			 col,
																			 removed_cnt,
																			 v,
																			 as_percent(removed_cnt / initial_cnt),
																			 BCOLORS.ENDC))

		removed_total = initial_cnt - cnt
		print('{}Total removed from \'{}\' {} out of {} values (remains: {}) = {}{}'.format(BCOLORS.C_GREEN,
																							col,
																							removed_total,
																							initial_cnt, cnt,
																							as_percent(removed_total / initial_cnt),
																							BCOLORS.ENDC))

	return df


def find_outliers_tukey(x):
	q1 = np.percentile(x, 25)
	q3 = np.percentile(x, 75)
	iqr = q3 - q1
	floor = q1 - 1.5 * iqr
	ceiling = q3 + 1.5 * iqr
	outlier_indices = list(x.index[(x < floor) | (x > ceiling)])
	outlier_values = list(x[outlier_indices])

	return outlier_indices, outlier_values


def remove_tukey_outliers(df, col):
	initial_cnt = len(df)

	outliers_ix, outliers_val = find_outliers_tukey(df[col])
	df = df.drop(outliers_ix)

	cnt = len(df)
	removed_total = initial_cnt - cnt

	print('{}Total removed from \'{}\' {} out of {} values (remains: {}) = {}{}'.format(BCOLORS.C_GREEN,
																						col,
																						removed_total,
																						initial_cnt, cnt,
																						as_percent(removed_total / initial_cnt),
																						BCOLORS.ENDC))
	print('Removed the following values: ', outliers_val)
	return df


def add_interactions(df, old_str=None, new_str=None):
	startTime = time.time()
	print('Start: ', startTime)

	# -- Get feature names
	combos = get_interaction_cols(df)
	col_names_new = ['_&_'.join(x) for x in combos]

	if old_str is not None:
		col_names_new = [x.replace(old_str, new_str) for x in col_names_new]

	col_names = list(df.columns) + col_names_new

	# -- Find interactions
	poly = PolynomialFeatures(interaction_only=True, include_bias=False)
	df = poly.fit_transform(df)
	df = pd.DataFrame(df)
	df.columns = col_names

	# -- Remove interaction terms with all 0 or 1 values
	# df = df.drop(df.columns[non_int_indices], axis=1)
	df = remove_all_0_1_cols(df)

	print('Duration: ', time.time() - startTime)
	return df


def get_interaction_cols(df):
	return list(combinations(list(df.columns), 2))


def remove_all_0_1_cols(df):
	df = df.drop(df.columns[(df == 0).all()], axis=1)
	df = df.drop(df.columns[(df == 1).all()], axis=1)
	return df


def add_cols_interactions(df, cols, old_str=None, new_str=None):
	cols_original = list(set(get_cols_alphabetically(df)) - set(cols))

	df_interactions = df[cols]
	df_interactions = add_interactions(df_interactions, old_str, new_str)
	df = pd.concat([df[cols_original], df_interactions], axis=1, join_axes=[df.index])
	return df.fillna(0)


def get_xy_data(y_col, df, dummy_cols=[], values_to_remove=None):
	for col in df.columns:
		rows, total = get_unique_rows([col], df)
		missing = len(df) - total
		print(col, ' | missing nan : ', missing)

		if missing > 0:
			print(BCOLORS.WARNING + 'WARNING: missing y value for x_y model in column > ' + col + BCOLORS.ENDC)
			break

	if len(dummy_cols) > 0:
		# get_cols_unique_counts(dummy_cols, df)
		df = dummy_df(df, dummy_cols)

	if values_to_remove:
		df = remove_values(values_to_remove, y_col, df)

	y = df[y_col]
	X = np.array(df.drop([y_col], 1))

	return X, y, df


def get_xy_cols(y_col, df):
	for col in df.columns:
		rows, total = get_unique_rows([col], df)
		missing = len(df) - total
		print(col, ' | missing nan : ', missing)

		if missing > 0:
			print(BCOLORS.WARNING + 'WARNING: missing y value for x_y model in column > ' + col + BCOLORS.ENDC)
			break

	y = df[y_col]
	X = drop_columns([y_col], df)
	cols = X.columns.tolist()

	return np.array(X), y, cols


def get_x_data(dummy_cols, df):
	for col in df.columns:
		rows, total = get_unique_rows([col], df)
		missing = len(df) - total
		print(col, rows, ' | missing nan : ', missing)

		if missing > 0:
			print(BCOLORS.WARNING + 'WARNING: missing y value for x_y model in column > ' + col + BCOLORS.ENDC)
			break

	if len(dummy_cols) > 0:
		# get_cols_unique_counts(dummy_cols, df)
		df = dummy_df(df, dummy_cols)

	return df


# endregion


# region GRAPHICS
# ------------------------------------------------------------------------------------------

def plot_histogram(x):
	plt.hist(x, color='gray', alpha=0.5)
	plt.title("Histogram of '{var_name}'".format(var_name=x.name))
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.show()


def plot_histogram_dv(x, y):
	plt.hist(list(x[y == 0]), alpha=0.5, label='Outcome=0')
	plt.hist(list(x[y == 1]), alpha=0.5, label='Outcome=1')
	plt.title("Histogram of '{var_name}' by Outcome Category".format(var_name=x.name))
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.legend(loc='upper right')
	plt.show()


def get_object_methods(o):
	return [method for method in dir(o) if callable(getattr(o, method))]


def get_region_palette():
	col_bxl = "#FFEB17"
	col_vl = "#006A8D"
	col_wal = "#D91302"
	reg_ui = [col_bxl, col_vl, col_wal]
	sns.set_palette(reg_ui)
	pal = sns.color_palette()
	return pal


def graph_pca_spree(x_data, min_comps, max_comps):
	pca = PCA()
	# pca = decomposition.KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
	plt.figure()
	plt.clf()
	# -- pca analysis : choosing number of factors
	for i in np.arange(min_comps, max_comps):
		pca = PCA(n_components=i)
		pca.fit(x_data)
		pca.explained_variance_ratio_
		plt.plot(pca.explained_variance_, linewidth=2)
	plt.axis('tight')
	plt.xlabel('n_components')
	plt.ylabel('explained_variance_')


def graph_cluster_data_with_knn(x_data, title, min_clusters=3, max_clusters=7):
	# ----------------------------------------------------------------------
	print(BCOLORS.C_RED, 'Clustering data', BCOLORS.ENDC)
	range_n_clusters = np.arange(min_clusters, max_clusters)

	for n_clusters in range_n_clusters:
		plot_clusters(x_data, n_clusters, title)


def plot_clusters(x_data, n_clusters, title):
	# Create a subplot with 1 row and 2 columns
	fig, (ax1, ax2) = plt.subplots(1, 2)
	fig.set_size_inches(18, 7)

	# The 1st subplot is the silhouette plot
	# The silhouette coefficient can range from -1, 1 but in this example all
	# lie within [-0.1, 1]
	ax1.set_xlim([-0.1, 1])

	# The (n_clusters+1)*10 is for inserting blank space between silhouette
	# plots of individual clusters, to demarcate them clearly.
	ax1.set_ylim([0, len(x_data) + (n_clusters + 1) * 10])

	# Initialize the clusterer with n_clusters value and a random generator
	# seed of 10 for reproducibility.
	clusterer = KMeans(n_clusters=n_clusters,
					   random_state=42,
					   n_init=100,
					   precompute_distances=True,
					   tol=1e-6,
					   init='random')
	cluster_labels = clusterer.fit_predict(x_data)

	# The silhouette_score gives the average value for all the samples.
	# This gives a perspective into the density and separation of the formed
	# clusters
	silhouette_avg = silhouette_score(x_data, cluster_labels)

	print("For n_clusters =", n_clusters,
		  "The average silhouette_score is :", silhouette_avg)

	# Compute the silhouette scores for each sample
	sample_silhouette_values = silhouette_samples(x_data, cluster_labels)
	y_lower = 10

	for i in range(n_clusters):
		# ----------------------------------------------------------------------
		print_time(TASK.start)

		# Aggregate the silhouette scores for samples belonging to
		# cluster i, and sort them
		ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

		ith_cluster_silhouette_values.sort()

		size_cluster_i = ith_cluster_silhouette_values.shape[0]
		y_upper = y_lower + size_cluster_i

		color = cm.spectral(float(i + 1) / n_clusters)
		ax1.fill_betweenx(np.arange(y_lower, y_upper),
						  0, ith_cluster_silhouette_values,
						  facecolor=color, edgecolor=color, alpha=0.7)

		# Label the silhouette plots with their cluster numbers at the middle
		ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

		# Compute the new y_lower for next plot
		y_lower = y_upper + 10  # 10 for the 0 samples

		# ----------------------------------------------------------------------
		print_time(TASK.end)

	ax1.set_title("qsdfsdf- The silhouette plot for the various clusters.")
	ax1.set_xlabel("The silhouette coefficient values")
	ax1.set_ylabel("Cluster label")

	# The vertical line for average silhoutte score of all the values
	ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
	ax1.set_yticks([])  # Clear the yaxis labels / ticks
	ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

	# 2nd Plot showing the actual clusters formed
	colors = cm.spectral((cluster_labels.astype(float) + 1) / n_clusters)
	ax2.scatter(x_data[:, 0], x_data[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)

	# Labeling the clusters
	centers = clusterer.cluster_centers_
	# Draw white circles at cluster centers
	ax2.scatter(centers[:, 0], centers[:, 1],
				marker='o', c="white", alpha=.95, s=400)

	for i, c in enumerate(centers):
		ax2.scatter(c[0], c[1], marker='$%d$' % (i + 1), alpha=1, s=50)

	ax2.set_title("The visualization of the clustered data.")
	ax2.set_xlabel("Feature space for the 1st feature")
	ax2.set_ylabel("Feature space for the 2nd feature")
	plt.suptitle('Silhouette analysis (KMeans clustering) on {} with n_clusters = {}'.format(title, n_clusters), fontsize=14, fontweight='bold')
	plt.show()
	png_file_to_save = graph_clusters_folder + '{} - {} clusters.png'.format(title, n_clusters)
	print(png_file_to_save)
	fig.savefig(png_file_to_save, dpi=300)
	plt.close(fig)

	return cluster_labels, centers


def box_plot_summary(category, clrs, df_x, nb_clusters, var_name, var_french):
	grp = 'Groupements'

	fig, ax = plt.subplots()
	sns.set_palette(clrs)
	sns.boxplot(x='Cluster', y=var_name, data=df_x, ax=ax)
	sns.despine(offset=15, trim=True, bottom=True)
	plt.suptitle(category, fontsize=16, fontweight='bold')
	ax.set_xlabel(grp)
	ax.set_ylabel(var_french)
	fig.tight_layout(h_pad=.9)
	plt.subplots_adjust(top=.9)
	plt.show()

	png_file_to_save = graph_clusters_folder + '{} - {} - Boxplot ({} {}).png'.format(category, var_name, nb_clusters, grp)
	print(png_file_to_save)
	plt.savefig(png_file_to_save, dpi=300)
	plt.close()


def graph_knn_clusters(cols_x, df_x, df_y, cat):
	n_comps = 2
	cats = df_y[cat].unique()
	min_clusters = 3
	max_clusters = 15

	scaler = MinMaxScaler()
	pca = PCA(n_comps, random_state=42)

	# -- Cluster
	for title in cats:
		print(title)
		if title is not None:
			df_x_selected = df_x[df_x[cat] == title]
			df_x_selected = df_x_selected[cols_x]

			if len(df_x_selected) > max_clusters:
				X = np.array(df_x_selected)
				X_scaled = scaler.fit_transform(X)
				X_pca = pca.fit_transform(X_scaled)

				df_cmps = DataFrame(pca.components_)
				df_cmps = df_cmps.set_index(np.arange(1, len(df_cmps) + 1))
				df_cmps.columns = df_x_selected.columns

				save_as_xlsx(df_cmps.transpose(), 'All factors for {} ({})'.format(title, n_comps))

				cols_kept = df_cmps[(df_cmps.abs() > .05)].dropna(axis=1, how='all').columns

				df_cmps = df_cmps[cols_kept].transpose()
				# df_cmps = df_cmps[(df_cmps.iloc[:, :2].abs() > .01).all(axis=1)]
				df_cmps['IMPORTANCE'] = abs(df_cmps[1] - df_cmps[2])

				df_cmps = df_cmps.sort_values(by='IMPORTANCE', ascending=False)

				save_as_xlsx(df_cmps, 'Main factors for {} - NISS ({})'.format(title, n_comps))

				graph_cluster_data_with_knn(X_pca, title, min_clusters, max_clusters)


def get_clusters_description(cols_x, df, df_y, col):
	# df = df_x
	# col = DV.prog_type_txt_cat
	n_comps = 2

	scaler = MinMaxScaler()
	pca = PCA(n_comps)

	# -- select variable to analyse
	categories = df_y[col].dropna().unique()
	categories = sorted(categories)
	get_unique_rows([col], df)

	for category in categories:
		if category is not None:
			category = 'Hôpital Psychatrique'
			df_x = df[df[col] == category]
			df_x = df_x[cols_x]

			# get_cols_alphabetically(df_x)

			if len(df_x) > n_comps + 1:
				nb_clusters = input('Number of clusters in category: \'{}\' ?'.format(category))
				# nb_clusters = 5
				print(nb_clusters, category)

				X = np.array(df_x)
				X_scaled = scaler.fit_transform(X)
				X_pca = pca.fit_transform(X_scaled)
				int_nb_clusters = int(nb_clusters)
				clusters, centers = plot_clusters(X_pca, int_nb_clusters, category)
				df_clusters = DataFrame(scaler.inverse_transform(pca.inverse_transform(centers)), columns=df_x.columns)
				df_clusters = df_clusters.set_index(np.arange(1, len(df_clusters) + 1))
				save_as_xlsx(df_clusters.transpose().sort_index(), '{} - {} centres'.format(category, nb_clusters))

				df_x['Cluster'] = clusters + 1

				clrs = [cm.spectral(x / int_nb_clusters) for x in np.arange(1, int_nb_clusters + 1)]

				box_plot_summary(category, clrs, df_x, nb_clusters, 'AGE_NUM', 'Age')
				# box_plot_summary(category, clrs, df_x, nb_clusters, DV.education_level_ord, "Niveau d'éducation")

				cd = df_clusters.describe().unstack()
				cd = DataFrame(cd, columns=['ALL'])

				for i, c in enumerate(centers):
					print(i)
					cluster_description = df_x[df_x['Cluster'] == i + 1].describe()
					cluster_description = DataFrame(cluster_description.unstack(), columns=[str(i + 1)])
					cluster_description.index
					cd = pd.concat([cd, cluster_description], axis=1)

				save_as_xlsx(cd, '{} - {} clusters'.format(category, nb_clusters))


def get_clusters_radar(cols_x, df, df_y, col):
	df = df_x
	col = DV.prog_type_txt_cat
	n_comps = 2

	scaler = MinMaxScaler()
	pca = PCA(n_comps)

	# -- select variable to analyse
	categories = df_y[col].dropna().unique()
	categories = sorted(categories)
	get_unique_rows([col], df)

	for category in categories:
		if category is not None:
			# TODO: testing hack
			category = 'Consultations spécialisées'

			df_x = df[df[col] == category]
			df_x = df_x[cols_x]

			# get_cols_alphabetically(df_x)

			if len(df_x) > n_comps + 1:
				nb_clusters = input('Number of clusters in category: \'{}\' ?'.format(category))
				# nb_clusters = 5
				print(nb_clusters, category)

				X = np.array(df_x)
				X_scaled = scaler.fit_transform(X)
				X_pca = pca.fit_transform(X_scaled)
				int_nb_clusters = int(nb_clusters)
				clusters, centers = plot_clusters(X_pca, int_nb_clusters, category)

				df_clusters = DataFrame(scaler.inverse_transform(pca.inverse_transform(centers)), columns=df_x.columns)
				df_clusters = df_clusters.set_index(np.arange(1, len(df_clusters) + 1))

				scaler_result = MinMaxScaler()

				df_report = DataFrame(scaler_result.fit_transform(df_clusters.copy()), columns=df_clusters.columns)
				df_report = df_report.transpose().sort_index()
				save_as_xlsx(df_report, '{} - {} centres'.format(category, nb_clusters))

				df_x['Cluster'] = clusters + 1

				clrs = [cm.spectral(x / int_nb_clusters) for x in np.arange(1, int_nb_clusters + 1)]

				box_plot_summary(category, clrs, df_x, nb_clusters, 'AGE_NUM', 'Age')
				# box_plot_summary(category, clrs, df_x, nb_clusters, DV.education_level_ord, "Niveau d'éducation")

				cd = df_clusters.describe().unstack()
				cd = DataFrame(cd, columns=['ALL'])

				for i, c in enumerate(centers):
					print(i)
					cluster_description = df_x[df_x['Cluster'] == i + 1].describe()
					cluster_description = DataFrame(cluster_description.unstack(), columns=[str(i + 1)])
					cluster_description.index
					cd = pd.concat([cd, cluster_description], axis=1)

				save_as_xlsx(cd, '{} - {} clusters'.format(category, nb_clusters))


# endregion

# region TIME

def print_time(task_name='Start'):
	print(task_name, current_fulltime())


def current_fulltime():
	return time.strftime("%a, %d %b %Y %H:%M:%S +0000")


class TASK:
	start = 'START'
	end = 'END'


# endregion
