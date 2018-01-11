import os

import yaml


class Config:
    def __init__(self):
        self._root_dir = os.path.dirname(__file__)
        config_file = os.path.join(self._root_dir, "config.yml")
        with open(config_file, 'r') as ymlfile:
            self._cfg = yaml.load(ymlfile)

    @property
    def concordance_file(self):
        class_dir = self._cfg.get('classifications_data_dir')
        concordance_fn = self._cfg.get('concordance_filename')
        return os.path.join(self._root_dir, class_dir, concordance_fn)

    @property
    def environmental_extensions_file(self):
        data_dir = self._cfg.get('data_dir')
        env_fn = self._cfg.get('env_extensions_filename')
        return os.path.join(self._root_dir, data_dir, env_fn)

    @property
    def cif_fob_categories_range(self):
        return self._cfg.get('cif_fob_categories_range')

    @property
    def direct_purchases_abroad_categories_range(self):
        return self._cfg.get('direct_purchases_abroad_categories_range')

    @property
    def tax_categories_range(self):
        return self._cfg.get('tax_categories_range')

    @property
    def purchases_non_residents_categories_range(self):
        return self._cfg.get('purchases_non_residents_categories_range')

    @property
    def product_categories_range(self):
        return self._cfg.get('product_categories_range')

    @property
    def industry_categories_range(self):
        return self._cfg.get('industry_categories_range')

    @property
    def finaluse_categories_range(self):
        return self._cfg.get('finaluse_categories_range')

    @property
    def value_added_categories_range(self):
        return self._cfg.get('value_added_categories_range')

    @property
    def env_extensions_categories_range(self):
        return self._cfg.get('env_extensions_categories_range')

    @property
    def supply_range(self):
        return self._cfg.get('supply_range')

    @property
    def domestic_use_range(self):
        return self._cfg.get('domestic_use_range')

    @property
    def domestic_final_use_range(self):
        return self._cfg.get('domestic_final_use_range')

    @property
    def import_use_range(self):
        return self._cfg.get('import_use_range')

    @property
    def final_import_use_range(self):
        return self._cfg.get('final_import_use_range')

    @property
    def cif_fob_range(self):
        return self._cfg.get('cif_fob_range')

    @property
    def tax_range(self):
        return self._cfg.get('tax_range')

    @property
    def final_tax_range(self):
        return self._cfg.get('final_tax_range')

    @property
    def direct_purchases_abroad_range(self):
        return self._cfg.get('direct_purchases_abroad_range')

    @property
    def purchases_non_residents_range(self):
        return self._cfg.get('purchases_non_residents_range')

    @property
    def value_added_range(self):
        return self._cfg.get('value_added_range')

    @property
    def environmental_extensions_range(self):
        return self._cfg.get('env_extensions_range')

    @property
    def concordance_range(self):
        return self._cfg.get('concordance_range')

    @property
    def final_use_category_indices(self):
        indices = []
        final_use_categories = self._cfg.get('final_use_categories')
        for category in final_use_categories:
            indices.append(category[0])
        return indices

    @property
    def value_added_indices(self):
        indices = []
        value_added_categories = self._cfg.get('value_added_categories')
        for category in value_added_categories:
            indices.append(category[0])
        return indices

    @property
    def base_year(self):
        return self._cfg.get('base_year')

    @property
    def countries(self):
        return list(self._cfg.get('countries'))

    @property
    def country_names(self):
        names = []
        countries = self._cfg.get('countries')
        for cntr in countries:
            names.append(cntr[0])
        return names

    @property
    def country_codes(self):
        codes = []
        countries = self._cfg.get('countries')
        for cntr in countries:
            codes.append(cntr[1])
        return codes

    @property
    def substance_names(self):
        names = []
        substances = self._cfg.get('substances')
        for subst in substances:
            names.append(subst[0])
        return names

    @property
    def substance_units(self):
        units = []
        substances = self._cfg.get('substances')
        for subst in substances:
            units.append(subst[1])
        return units

    @property
    def data_dir(self):
        data_dir = self._cfg.get('data_dir')
        return os.path.join(self._root_dir, data_dir)

    @property
    def results_dir(self):
        results_dir = self._cfg.get('results_dir')
        return os.path.join(self._root_dir, results_dir)

    @property
    def classifications_file(self):
        class_fn = self._cfg.get('classifications_filename')
        class_dir = self._cfg.get('classifications_data_dir')
        return os.path.join(self._root_dir, class_dir, class_fn)

    @property
    def product_count(self):
        return self._cfg.get('product_count')

    @property
    def industry_count(self):
        return self._cfg.get('industry_count')

    @property
    def finaluse_count(self):
        return self._cfg.get('finaluse_count')

    @property
    def value_added_count(self):
        return self._cfg.get('value_added_count')

    @property
    def tax_count(self):
        return self._cfg.get('tax_count')

    @property
    def cif_fob_count(self):
        return self._cfg.get('cif_fob_count')

    @property
    def direct_purchases_abroad_count(self):
        return self._cfg.get('direct_purchases_count')

    @property
    def purchases_non_residents_count(self):
        return self._cfg.get('purchases_non_residents_count')
