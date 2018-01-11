import numpy as np
import config as cfg


class Sut:
    """A data transfer object that contains data from one supply-use table."""

    def __init__(self):
        self._cntrcode = None
        self._year = None
        self._supply = None
        self._domestic_use = None
        self._domestic_final_use = None
        self._import_use = None
        self._import_final_use = None
        self._value_added = None
        self._tax = None
        self._final_tax = None
        self._cif_fob = None
        self._direct_purchases_abroad = None
        self._purchases_non_residents = None
        self._product_categories = None
        self._industry_categories = None
        self._finaluse_categories = None
        self._value_added_categories = None
        self._cif_fob_categories = None
        self._tax_categories = None
        self._direct_purchases_abroad_categories = None
        self._purchases_non_residents_categories = None
        self._settings = cfg.Config()

    @property
    def product_categories(self):
        return self._product_categories

    @property
    def industry_categories(self):
        return self._industry_categories

    @property
    def finaluse_categories(self):
        return self._finaluse_categories

    @property
    def value_added_categories(self):
        return self._value_added_categories

    @product_categories.setter
    def product_categories(self, categories):
        assert type(categories) is list
        assert len(categories) is self._settings.product_count
        self._product_categories = categories

    @industry_categories.setter
    def industry_categories(self, categories):
        assert type(categories) is list
        assert len(categories) is self._settings.industry_count
        self._industry_categories = categories

    @finaluse_categories.setter
    def finaluse_categories(self, categories):
        assert type(categories) is list
        assert len(categories) is self._settings.finaluse_count
        self._finaluse_categories = categories

    @value_added_categories.setter
    def value_added_categories(self, categories):
        assert type(categories) is list
        assert len(categories) is self._settings.value_added_count
        self._value_added_categories = categories

    @property
    def cif_fob_categories(self):
        return self._cif_fob_categories

    @property
    def tax_categories(self):
        return self._tax_categories

    @property
    def direct_purchases_abroad_categories(self):
        return self._direct_purchases_abroad_categories

    @property
    def purchases_non_residents_categories(self):
        return self._purchases_non_residents_categories

    @cif_fob_categories.setter
    def cif_fob_categories(self, categories):
        assert type(categories) is list
        assert len(categories) is self._settings.cif_fob_count
        self._cif_fob_categories = categories

    @tax_categories.setter
    def tax_categories(self, categories):
        assert type(categories) is list
        assert len(categories) is self._settings.tax_count
        self._tax_categories = categories

    @direct_purchases_abroad_categories.setter
    def direct_purchases_abroad_categories(self, categories):
        assert type(categories) is list
        assert len(categories) is self._settings.direct_purchases_abroad_count
        self._direct_purchases_abroad_categories = categories

    @purchases_non_residents_categories.setter
    def purchases_non_residents_categories(self, categories):
        assert type(categories) is list
        assert len(categories) is self._settings.purchases_non_residents_count
        self._purchases_non_residents_categories = categories

    @property
    def cntr_code(self):
        return self._cntrcode

    @cntr_code.setter
    def cntr_code(self, code):
        assert type(code) is str
        assert len(code) == 2
        self._cntrcode = code

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, yr):
        assert type(yr) is int
        self._year = yr

    @property
    def supply(self):
        return self._supply

    @supply.setter
    def supply(self, sup):
        assert type(sup) is np.ndarray
        assert sup.dtype == np.float64
        assert sup.shape == (self._settings.product_count, self._settings.industry_count)
        self._supply = sup

    @property
    def use(self):
        return self._domestic_use + self._import_use

    @property
    def final_use(self):
        return self._domestic_final_use + self._import_final_use

    @property
    def import_use(self):
        return self._import_use

    @import_use.setter
    def import_use(self, imp_use):
        assert type(imp_use) is np.ndarray
        assert imp_use.dtype == np.float64
        assert imp_use.shape == (self._settings.product_count, self._settings.industry_count)
        self._import_use = imp_use

    @property
    def import_final_use(self):
        return self._import_final_use

    @import_final_use.setter
    def import_final_use(self, imp_final_use):
        assert type(imp_final_use) is np.ndarray
        assert imp_final_use.dtype == np.float64
        assert imp_final_use.shape == (self._settings.product_count, self._settings.finaluse_count)
        self._import_final_use = imp_final_use

    @property
    def domestic_use(self):
        return self._domestic_use

    @domestic_use.setter
    def domestic_use(self, dom_use):
        assert type(dom_use) is np.ndarray
        assert dom_use.dtype == np.float64
        assert dom_use.shape == (self._settings.product_count, self._settings.industry_count)
        self._domestic_use = dom_use

    @property
    def domestic_final_use(self):
        return self._domestic_final_use

    @domestic_final_use.setter
    def domestic_final_use(self, dom_final_use):
        assert type(dom_final_use) is np.ndarray
        assert dom_final_use.dtype == np.float64
        assert dom_final_use.shape == (self._settings.product_count, self._settings.finaluse_count)
        self._domestic_final_use = dom_final_use

    @property
    def total_product_import_use(self):
        return np.sum(self._import_use, axis=1) + np.sum(self._import_final_use, axis=1)

    @property
    def total_product_domestic_use(self):
        return np.sum(self._domestic_use, axis=1) + np.sum(self._domestic_final_use, axis=1)

    @property
    def total_product_use(self):
        return (np.sum(self._domestic_use, axis=1) + np.sum(self._import_use, axis=1) +
                np.sum(self._domestic_final_use, axis=1) + np.sum(self._import_final_use, axis=1))

    @property
    def total_product_supply(self):
        return (np.sum(self._supply, axis=1) + np.sum(self._import_use, axis=1)
                + np.sum(self._import_final_use, axis=1))

    @property
    def total_industry_output(self):
        return np.sum(self._supply, axis=0)

    @property
    def total_industry_input(self):
        return (np.sum(self._domestic_use, axis=0) + np.sum(self._import_use, axis=0) +
                np.sum(self._value_added, axis=0) + np.sum(self._tax, axis=0))

    @property
    def value_added(self):
        return self._value_added

    @value_added.setter
    def value_added(self, va):
        assert type(va) is np.ndarray
        assert va.dtype == np.float64
        assert va.shape == (self._settings.value_added_count, self._settings.industry_count)
        self._value_added = va

    @property
    def tax(self):
        return self._tax

    @tax.setter
    def tax(self, taxes):
        assert type(taxes) is np.ndarray
        assert taxes.dtype == np.float64
        assert taxes.shape == (self._settings.tax_count, self._settings.industry_count)
        self._tax = taxes

    @property
    def final_tax(self):
        return self._final_tax

    @final_tax.setter
    def final_tax(self, final_taxes):
        assert type(final_taxes) is np.ndarray
        assert final_taxes.dtype == np.float64
        assert final_taxes.shape == (self._settings.tax_count, self._settings.finaluse_count)
        self._final_tax = final_taxes

    @property
    def cif_fob(self):
        return self._cif_fob

    @cif_fob.setter
    def cif_fob(self, cif_fob_adj):
        assert type(cif_fob_adj) is np.ndarray
        assert cif_fob_adj.dtype == np.float64
        assert cif_fob_adj.shape == (self._settings.cif_fob_count, self._settings.finaluse_count)
        self._cif_fob = cif_fob_adj

    @property
    def direct_purchases_abroad(self):
        return self._direct_purchases_abroad

    @direct_purchases_abroad.setter
    def direct_purchases_abroad(self, direct_purchases):
        assert type(direct_purchases) is np.ndarray
        assert direct_purchases.dtype == np.float64
        assert direct_purchases.shape == (self._settings.direct_purchases_abroad_count, self._settings.finaluse_count)
        self._direct_purchases_abroad = direct_purchases

    @property
    def purchases_non_residents(self):
        return self._purchases_non_residents

    @purchases_non_residents.setter
    def purchases_non_residents(self, non_resident_purchases):
        assert type(non_resident_purchases) is np.ndarray
        assert non_resident_purchases.dtype == np.float64
        assert non_resident_purchases.shape == (self._settings.purchases_non_residents_count,
                                                self._settings.finaluse_count)
        self._purchases_non_residents = non_resident_purchases

    @property
    def total_final_uses(self):
        return (np.sum(self._domestic_final_use, axis=0) + np.sum(self._import_final_use, axis=0) +
                self._cif_fob + self._purchases_non_residents + self._direct_purchases_abroad + self._final_tax)
