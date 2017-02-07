from utils import *


class DV:
	# df = read_pickle('Detailed_Institutions')
	# cols = get_cols_alphabetically(df)
	# for col in cols:
	# 	print('{} = \'{}\''.format(col.lower(), col))

	acces = 'ACCES'
	activfr = 'ACTIVFR'
	adnum = 'ADNUM'
	adresfr = 'ADRESFR'
	agrement = 'AGREMENT'
	benef = 'BENEF'
	bnum = 'BNUM'
	but = 'BUT'
	categ = 'CATEG'
	commune = 'COMMUNE'
	email = 'EMAIL'
	email_nl = 'EMAIL_NL'
	fax = 'FAX'
	fiche = 'FICHE'
	http = 'HTTP'
	http_nl = 'HTTP_NL'
	langstat = 'LANGSTAT'
	latitude = 'LATITUDE'
	longitude = 'LONGITUDE'
	mother = 'MOTHER'
	natoffre = 'NATOFFRE'
	nmofffr = 'NMOFFFR'
	nmservfr = 'NMSERVFR'
	nmusefr = 'NMUSEFR'
	offrling = 'OFFRLING'
	permanfr = 'PERMANFR'
	postfr = 'POSTFR'
	remarque = 'REMARQUE'
	revue = 'REVUE'
	section = 'SECTION'
	section_id = 'SECTION_id'
	statut = 'STATUT'
	statuut = 'STATUUT'
	subject = 'SUBJECT'
	subject_id = 'SUBJECT_id'
	tel = 'TEL'
	tel_nl = 'TEL_NL'
	terrain = 'TERRAIN'
	theme = 'THEME'
	topic = 'TOPIC'
	topic_id = 'TOPIC_id'
	xcoord = 'XCOORD'
	xcoord_th = 'XCOORD_TH'
	ycoord = 'YCOORD'
	ycoord_th = 'YCOORD_TH'
	zip = 'ZIP'


def fiches_as_pickle():
	df = pd.read_excel(excel_folder + 'CDCS_fiches.xls')
	df.columns

	# 'PM_RP', idendification registre national entreprise

	cols = ['ACCES',
			'ADNUM',
			'ADRESFR',
			'AGREMENT',
			'ACTIVFR',
			'BENEF',
			'BNUM',
			'BUT',
			'CATEG',
			'COMMUNE',
			'EMAIL',
			'EMAIL_NL',
			'FAX',
			'FICHE',
			'HTTP',
			'HTTP_NL',
			'LANGSTAT',
			'LATITUDE',
			'LONGITUDE',
			'MOTHER',
			'NATOFFRE',
			'NMOFFFR',
			'NMSERVFR',
			'NMUSEFR',
			'OFFRLING',
			'PERMANFR',
			'POSTFR',
			'REMARQUE',
			'REVUE',
			'STATUT',
			'STATUUT',
			'TEL',
			'TEL_NL',
			'TERRAIN',
			'THEME',
			'XCOORD',
			'XCOORD_TH',
			'YCOORD',
			'YCOORD_TH',
			'ZIP']

	df = df[cols]

	get_cols_alphabetically(df)

	df.loc[df[DV.xcoord] == 0 | df[DV.xcoord].isnull(), 'LATITUDE'] = 0
	df.loc[df[DV.xcoord] == 0 | df[DV.xcoord].isnull(), 'LONGITUDE'] = 0

	df.dropna(subset=[DV.categ], inplace=True)
	df[DV.categ] = df[DV.categ]
	save_as_pickle(df, 'Institutions')


# def categories_as_pickle():
# 	df_categories = pd.read_excel(excel_folder + 'CDCS_cats.xlsx')
# 	df_categories.columns
# 	df_categories = df_categories.fillna(999)
# 
# 	df_categories.SUB_id = df_categories.SUB_id.astype(int)
# 	df_categories.MAIN_id = df_categories.MAIN_id.astype(int)
# 	save_as_pickle(df_categories, 'Categories')


def categories():
	df = pd.read_excel(excel_folder + 'CDCS_SECTIONS.xlsx')
	df = df.dropna(how='all')

	df.loc[df[DV.topic_id].isnull() & df[DV.topic_id].shift(-1) > 0, DV.subject_id] = 1

	new_ids = DataFrame(df[df[DV.subject_id] > 0][DV.subject_id])
	new_ids[DV.subject_id] = np.arange(1, new_ids.size + 1)
	del df[DV.subject_id]
	df = pd.concat([df, new_ids], axis=1)

	df.loc[df[DV.section].notnull(), DV.section_id] = 1

	new_ids = DataFrame(df[df[DV.section_id] > 0][DV.section_id])
	new_ids[DV.section_id] = np.arange(1, new_ids.size + 1)
	del df[DV.section_id]
	df = pd.concat([df, new_ids], axis=1)

	df.loc[df[DV.topic_id].isnull() & df[DV.subject_id] > 0, 'SUBJECT'] = df['TOPIC']
	df.loc[df[DV.topic_id].isnull() & df[DV.subject_id] > 0, 'TOPIC'] = np.nan

	df = df.dropna(how='all')
	df = df.fillna(method='ffill')
	print_full_rows(df, 10)

	save_as_pickle(df, 'Categories')


def detailed_institutions_as_pickle():
	# DV.category = "CATÉGORIE"
	# DV.maincategory = "CATÉGORIE PRINCIPALE"
	df = read_pickle('Institutions')
	get_cols_alphabetically(df)

	df_categories = df[DV.categ].str.split(',').apply(pd.Series, 1).stack()
	df_categories.dropna()
	df_categories.index = df_categories.index.droplevel(-1)
	df_categories.name = DV.categ

	del df[DV.categ]
	get_cols_alphabetically(df)

	df = df.join(df_categories)
	df[DV.categ] = df[DV.categ].str.strip()
	df = df[df[DV.categ] != '']
	df[DV.categ] = df[DV.categ].astype(int)
	df.dropna(subset=[DV.categ])

	get_unique_values(DV.categ, df)

	df_categories = read_pickle('Categories')
	df_categories = df_categories.dropna()
	df_categories[df_categories[DV.topic_id] == 482]

	df = pd.merge(df, df_categories, left_on=DV.categ, right_on=DV.topic_id)

	save_as_pickle(df, 'Detailed_Institutions')


# save_as_xlsx(df, 'Detailed_Institutions')


def check_assuetudes_coords():
	df = read_pickle('Detailed_Institutions')
	get_unique_rows([DV.section, DV.section_id], df)
	df[df[DV.xcoord] == 0 | df[DV.xcoord].isnull()]


def get_assuetudes():
	DV.fulladr = "FULLADRESS"
	df = read_pickle('Detailed_Institutions')

	df[DV.fulladr] = df[DV.adresfr] + ' ' + df[DV.adnum].astype(str) + ', ' + df[DV.postfr]

	df_categories = read_pickle('Categories')
	df_categories = df_categories.dropna()

	sections = get_unique_rows([DV.section, DV.section_id], df)
	subjects = get_unique_rows([DV.subject, DV.subject_id], df)
	topics = get_unique_rows([DV.topic, DV.topic_id], df)

	cats = get_unique_rows([DV.section, DV.section_id, DV.subject, DV.subject_id, DV.topic, DV.topic_id], df)
	# save_as_xlsx(cats, 'sections')

	pvt = pd.pivot_table(cats, index=[DV.section, DV.subject])
	pvt.columns
	pvt.index

	cols = [DV.section,
			DV.section_id,
			DV.subject,
			DV.subject_id,
			DV.topic,
			DV.topic_id,
			DV.fiche,
			DV.nmofffr,
			DV.activfr,
			DV.agrement,
			DV.benef,
			DV.fulladr]

	cols_pvt = [DV.fiche,
				DV.nmofffr,
				DV.agrement,
				DV.activfr,
				DV.fulladr,
				DV.topic,
				DV.subject]

	# selected_cats = [18, 20, 16, 17, 4]
	selected_cats = [18]

	df_tmp = df[df[DV.section_id].isin(selected_cats)][cols]
	# print_full(df_tmp)

	df_pivot = pd.pivot_table(df_tmp, index=cols_pvt)

	df_pivot = df_pivot.reset_index(level=5)

	print_full(df_pivot)
	print_full(df_pivot.index.droplevel(-1))
	df
	df_pivot.unstack(level='TOPIC')
	df
	del df_pivot[DV.topic_id]
	del df_pivot[DV.section_id]
	del df_pivot[DV.subject_id]

	df_pivot.stack()
	print("this is a github testing change")
	save_as_xlsx(df_pivot, 'Institutions_tox')
