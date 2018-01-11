import numpy as np

import config as cfg
import sut as st


class SutChecker:

    positive_fd_category_idx = (0, 1, 2, 3, 5)

    # default is the use of absolute value difference of 1E-6
    # if both absolute and relative difference is given
    # priority is given to the absolute difference
    def __init__(self, sut, absolute=None, relative=None):
        assert type(sut) is st.Sut

        if absolute is None and relative is None:
            self._use_abs = True
            self._use_rel = False
            self._abs_diff = 1E-6
            self._rel_diff = None

        if absolute is not None and relative is None:
            self._use_abs = True
            self._use_rel = False
            self._abs_diff = absolute
            self._rel_diff = None

        if absolute is None and relative is not None:
            self._use_abs = False
            self._use_rel = True
            self._abs_diff = None
            self._rel_diff = relative

        if absolute is not None and relative is not None:
            self._use_abs = True
            self._use_rel = False
            self._abs_diff = absolute
            self._rel_diff = None

        self._sut = sut
        self._settings = cfg.Config()
        self._log = list()

    def logical_checks(self):
        self._check_logics_prd()
        self._check_logics_ind()
        self._check_for_negatives()

    def value_checks(self):
        self._check_value_prd()
        self._check_value_ind()

    @property
    def log(self):
        return self._log

    def _check_logics_prd(self):
        prd_supply = self._total_product_supply()
        prd_use = self._total_product_use()
        for idx in range(0, self._settings.product_count):
            if prd_supply[idx, 0] == 0 and prd_use[idx, 0] != 0:
                msg = 'inconsistency in product supply and use for: ' + self._sut.product_categories[idx] + \
                    ', supply = {:.9f}, use = {:.9f}'.format(prd_supply[idx, 0], prd_use[idx, 0])
                self._log.append(msg)
            elif prd_supply[idx, 0] != 0 and prd_use[idx, 0] == 0:
                msg = 'inconsistency in product supply and use for: ' + self._sut.product_categories[idx] + \
                      ', supply = {:.9f}, use = {:.9f}'.format(prd_supply[idx, 0], prd_use[idx, 0])
                self._log.append(msg)

    def _check_logics_ind(self):
        ind_supply = self._total_industry_supply()
        ind_use = self._total_industry_use()
        for idx in range(0, self._settings.industry_count):
            if ind_supply[0, idx] == 0 and ind_use[0, idx] != 0:
                msg = 'inconsistency in industry output and input for: ' + self._sut.industry_categories[idx] + \
                      ', output = {:.9f}, input =  {:.9f}'.format(ind_supply[0, idx], ind_use[0, idx])
                self._log.append(msg)
            elif ind_supply[0, idx] != 0 and ind_use[0, idx] == 0:
                msg = 'inconsistency in industry output and input for: ' + self._sut.industry_categories[idx] + \
                      ', output = {:.9f}, input =  {:.9f}'.format(ind_supply[0, idx], ind_use[0, idx])
                self._log.append(msg)

    def _check_for_negatives(self):
        sup = self._sut.supply
        cnt = np.size(sup[sup < 0.0])
        if not cnt == 0:
            msg = '{} negative value(s) detected in supply table'.format(cnt)
            self._log.append(msg)

        dom = self._sut.domestic_use
        cnt = np.size(dom[dom < 0.0])
        if not cnt == 0:
            msg = '{} negative value(s) detected in domestic use table'.format(cnt)
            self._log.append(msg)

        imp = self._sut.import_use
        cnt = np.size(imp[imp < 0.0])
        if not cnt == 0:
            msg = '{} negative value(s) detected in import use table'.format(cnt)
            self._log.append(msg)

        dom_fd = self._sut.domestic_final_use
        dom_fd = dom_fd[:, self.positive_fd_category_idx]
        cnt = np.size(dom_fd[dom_fd < 0.0])
        if not cnt == 0:
            msg = '{} negative value(s) detected in domestic final demand table'.format(cnt)
            self._log.append(msg)

        imp_fd = self._sut.import_final_use
        imp_fd = imp_fd[:, self.positive_fd_category_idx]
        cnt = np.size(imp_fd[imp_fd < 0.0])
        if not cnt == 0:
            msg = '{} negative value(s) detected in import final demand table'.format(cnt)
            self._log.append(msg)

        tot_use = dom + imp
        cnt = np.size(tot_use[tot_use < 0.0])
        if not cnt == 0:
            msg = '{} negative value(s) detected in use table'.format(cnt)
            self._log.append(msg)

        tot_fd_use = dom_fd + imp_fd
        cnt = np.size(tot_fd_use[tot_fd_use < 0.0])
        if not cnt == 0:
            msg = '{} negative value(s) detected in final use table'.format(cnt)
            self._log.append(msg)

    def _check_value_ind(self):
        ind_supply = self._total_industry_supply()
        ind_use = self._total_industry_use()
        for idx in range(0, self._settings.industry_count):
            if ind_supply[0, idx] != 0:
                abs_diff = np.abs(ind_supply[0, idx] - ind_use[0, idx])
                rel_diff = 100 * abs_diff / ind_supply[0, idx]
                if self._use_abs is True and abs_diff > self._abs_diff:
                    msg = 'difference in industry output and input for: ' + self._sut.industry_categories[idx] + \
                          ', output = {:.9f}, input =  {:.9f}'.format(ind_supply[0, idx], ind_use[0, idx])
                    self._log.append(msg)
                if self._use_rel is True and rel_diff > self._rel_diff:
                    msg = 'difference in industry output and input for: ' + self._sut.industry_categories[idx] + \
                          ', output = {:.9f}, input =  {:.9f}'.format(ind_supply[0, idx], ind_use[0, idx])
                    self._log.append(msg)

    def _check_value_prd(self):
        prd_supply = self._total_product_supply()
        prd_use = self._total_product_use()
        for idx in range(0, self._settings.product_count):
            if prd_supply[idx, 0] != 0:
                abs_diff = np.abs(prd_supply[idx, 0] - prd_use[idx, 0])
                rel_diff = 100 * abs_diff / prd_supply[idx, 0]
                if self._use_abs is True and abs_diff > self._abs_diff:
                    msg = 'difference in product output and input for: ' + self._sut.product_categories[idx] + \
                          ', supply = {:.9f}, use = {:.9f}'.format(prd_supply[idx, 0],  prd_use[idx, 0])
                    self._log.append(msg)
                if self._use_rel is True and rel_diff > self._rel_diff:
                    msg = 'difference in product output and input for: ' + self._sut.product_categories[idx] + \
                          ', supply = {:.9f}, use = {:.9f}'.format(prd_supply[idx, 0], prd_use[idx, 0])
                    self._log.append(msg)

    def _total_product_supply(self):
        tot = np.sum(self._sut.supply, axis=1, keepdims=True)
        return tot

    def _total_product_use(self):
        tot = np.sum(self._sut.domestic_use, axis=1, keepdims=True) + \
              np.sum(self._sut.domestic_final_use, axis=1, keepdims=True)
        return tot

    def _total_industry_supply(self):
        tot = np.sum(self._sut.supply, axis=0, keepdims=True)
        return tot

    def _total_industry_use(self):
        tot_imp_use = np.sum(self._sut.import_use, axis=0, keepdims=True)
        tot_dom_use = np.sum(self._sut.domestic_use, axis=0, keepdims=True)
        tot_tax = np.sum(self._sut.tax, axis=0, keepdims=True)
        tot_va = np.sum(self._sut.value_added, axis=0, keepdims=True)
        tot = tot_imp_use + tot_dom_use + tot_tax + tot_va
        return tot
