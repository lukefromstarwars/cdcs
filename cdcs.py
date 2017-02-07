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
	section_id = 'SECTION_ID'
	statut = 'STATUT'
	statuut = 'STATUUT'
	subject = 'SUBJECT'
	subject_id = 'SUBJECT_ID'
	tel = 'TEL'
	tel_nl = 'TEL_NL'
	terrain = 'TERRAIN'
	theme = 'THEME'
	topic = 'TOPIC'
	topic_id = 'TOPIC_ID'
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


def rename_agreements():
	df = read_pickle('Institutions')

	old_str = ['Cfl', 'C.flamande', 'C.fla.', 'Cfl', 'Comm.flamande']
	new_str = 'Communauté flamande'

	# old_str = ['K&G']
	# new_str = 'Kind en Gezin'

	# old_str = ['Région Bruxelles-Capitale']
	# new_str = 'RBC'

	for str in old_str:
		df[DV.agrement] = df[DV.agrement].str.replace(str, new_str)

	# str_contains = df[DV.agrement].str.contains(old_str, case=False, na=False)
	# df[str_contains]
	#
	agreements_list = get_unique_rows([DV.agrement], df)
	print_full(agreements_list)

	# save_as_xlsx(agreements_list, 'Agreements')
	save_as_pickle(df, 'Institutions_renamed')


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
	# rename_agreements()

	DV.agrement_organisation = 'AGREMENT_ORGANISATION'

	df = read_pickle('Institutions')
	get_cols_alphabetically(df)

	# split categories
	split_str = ','
	df_categories = df[DV.categ].str.split(split_str).apply(pd.Series, 1).stack()
	df_categories.dropna()
	df_categories.index = df_categories.index.droplevel(-1)
	df_categories.name = DV.categ

	# split agreement
	split_str = '\\r\\n'
	df_agreement = df[DV.agrement].str.split(split_str).apply(pd.Series, 1).stack()
	df_agreement.dropna()
	df_agreement.index = df_agreement.index.droplevel(-1)
	df_agreement = DataFrame(df_agreement, columns=[DV.agrement])

	get_unique_rows([DV.agrement], df_agreement)

	print_full(df_agreement)

	# split agreement from organization
	split_str = ' - '
	df_agreement_with_orgs = df_agreement[DV.agrement].str.split(split_str).apply(pd.Series, 1)
	df_agreement_with_orgs.dropna(how='all')
	df_agreement_with_orgs.columns = [DV.agrement, DV.agrement_organisation]
	get_unique_rows([DV.agrement_organisation, DV.agrement], df_agreement_with_orgs)
	agreement_orgs = get_unique_rows([DV.agrement_organisation], df_agreement_with_orgs)

	# rename cols

	old_strs = [
		'Cfl',
		'C.flamande',
		'C.fla.',
		'Cfl',
		'Comm.flamande',
		'K&G',
		'RBC',
		'AUTORITÉ FÉDÉRALE'
	]

	new_strs = [
		'Communauté flamande',
		'Communauté flamande',
		'Communauté flamande',
		'Communauté flamande',
		'Communauté flamande',
		'Kind en Gezin',
		'Région Bruxelles-Capitale',
		'Fédéral'
	]

	for o, n in zip(old_strs, new_strs):
		print(o, n)
		df_agreement_with_orgs[DV.agrement_organisation] = df_agreement_with_orgs[DV.agrement_organisation].str.replace(o, n)

	df_agreement_with_orgs.replace(old_strs, new_strs, inplace=True)

	save_as_xlsx(agreement_orgs, 'Agreement_orgs')

	# merge detailed institutions
	del df[DV.categ]
	del df[DV.agrement]

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
	get_cols_alphabetically(df)

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
	agreement = get_unique_rows([DV.agrement], df)
	save_as_xlsx(agreement, 'Agreement')

	cats = get_unique_rows([DV.section, DV.section_id, DV.subject, DV.subject_id, DV.topic, DV.topic_id], df)
	# save_as_xlsx(cats, 'sections')

	pvt = pd.pivot_table(cats, index=[DV.section, DV.subject])
	pvt.columns
	pvt.index

	cols = [
		DV.activfr,
		DV.agrement,
		DV.benef,
		DV.fiche,
		DV.fulladr,
		DV.nmofffr,
		DV.offrling,
		DV.permanfr,
		DV.section_id,
		DV.subject,
		DV.subject_id,
		DV.topic,
		DV.topic_id,
		DV.section
	]

	cols_pvt = [
		DV.fiche,
		DV.nmofffr,
		DV.fulladr,
		DV.benef,
		DV.offrling,
		DV.permanfr,
		DV.agrement,
		DV.subject,
		DV.topic,
		DV.activfr
	]

	# selected_cats = [18, 20, 16, 17, 4]
	selected_cats = [18]

	df_tmp = df[df[DV.section_id].isin(selected_cats)][cols]
	# print_full(df_tmp)

	df_pivot = pd.pivot_table(df_tmp, index=cols_pvt)
	df_pivot.to_html(open(html_folder + 'new_file.html', 'w'))

	df_institutions = df_pivot.reset_index()

	df_pivot = df_pivot.reset_index()

	print_full(df_pivot)
	print_full(df_pivot.index.droplevel(-1))
	df_pivot.unstack(level='TOPIC')
	del df_pivot[DV.topic_id]
	del df_pivot[DV.section_id]
	del df_pivot[DV.subject_id]

	df_pivot.stack()

	save_as_xlsx(df_institutions, 'Institutions_tox')
