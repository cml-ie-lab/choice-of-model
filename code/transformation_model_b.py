import math

import numpy as np

import sut as st


class TransformationModelB:
    """A supply-use table to input-output table transformation object.
    From the supply-use table a product-by-product input-output table
    based on industry technology assumption is created. In the 'Eurostat Manual
    of Supply, Use and Input-Output Tables' this transformation model is called
    model B. The resulting input-output table does not contain negative values. Only
    the domestic tables are taken into consideration"""

    default_rel_tol = 1E-3

    def __init__(self, sut, env_extensions):
        assert type(sut) is st.Sut
        self._sut = sut
        self._ext = env_extensions

    def transformation_matrix(self):
        make = np.transpose(self._sut.supply)
        g = self._sut.total_industry_output
        return np.dot(_invdiag(g), make)

    def io_transaction_matrix(self):
        use = self._sut.domestic_use
        return np.dot(use, self.transformation_matrix())

    def io_coefficient_matrix(self):
        q = self._sut.total_product_domestic_use
        return np.dot(self.io_transaction_matrix(), _invdiag(q))

    def ext_transaction_matrix(self):
        return np.dot(self._ext, self.transformation_matrix())

    def ext_coefficients_matrix(self):
        q = self._sut.total_product_domestic_use
        return np.dot(self.ext_transaction_matrix(), _invdiag(q))

    def final_demand(self, fd=None):
        if fd is None:
            fd = self._sut.domestic_final_use
        return fd

    def check_io_transaction_matrix(self, rel_tol=default_rel_tol):
        is_correct = True
        q1 = np.sum(self.io_transaction_matrix(), axis=1) + \
            np.sum(self._sut.domestic_final_use, axis=1)
        q2 = np.sum(self._sut.domestic_use, axis=1) + \
            np.sum(self._sut.domestic_final_use, axis=1)
        it = np.nditer(q1, flags=['f_index'])
        while not it.finished and is_correct:
            if not math.isclose(q1[it.index], q2[it.index], rel_tol=rel_tol):
                is_correct = False
            it.iternext()
        return is_correct

    def check_io_coefficients_matrix(self, rel_tol=default_rel_tol):
        is_correct = True
        q1 = np.sum(self.io_transaction_matrix(), axis=1) + \
            np.sum(self._sut.domestic_final_use, axis=1)
        (row_cnt, col_cnt) = self._sut.use.shape
        eye = np.diag(np.ones(row_cnt))
        l_inverse = np.linalg.inv(eye - self.io_coefficient_matrix())
        fd = np.sum(self._sut.domestic_final_use, axis=1)
        q2 = np.dot(l_inverse, fd)
        it = np.nditer(q1, flags=['f_index'])
        while not it.finished and is_correct:
            if not math.isclose(q1[it.index], q2[it.index], rel_tol=rel_tol):
                is_correct = False
            it.iternext()
        return is_correct

    def check_ext_transaction_matrix(self, rel_tol=default_rel_tol):
        is_correct = True
        e1 = np.sum(self._ext, axis=1)
        e2 = np.sum(self.ext_transaction_matrix(), axis=1)
        it = np.nditer(e1, flags=['f_index'])
        while not it.finished and is_correct:
            if not math.isclose(e1[it.index], e2[it.index], rel_tol=rel_tol):
                is_correct = False
            it.iternext()
        return is_correct

    def check_ext_coefficient_matrix(self, rel_tol=default_rel_tol):
        is_correct = True
        e1 = np.sum(self._ext, axis=1)
        ext = self.ext_coefficients_matrix()
        (row_cnt, col_cnt) = self._sut.use.shape
        eye = np.diag(np.ones(row_cnt))
        l_inverse = np.linalg.inv(eye - self.io_coefficient_matrix())
        fd = np.sum(self._sut.domestic_final_use, axis=1)
        e2 = np.dot(ext, np.dot(l_inverse, fd))
        it = np.nditer(e1, flags=['f_index'])
        while not it.finished and is_correct:
            if not math.isclose(e1[it.index], e2[it.index], rel_tol=rel_tol):
                is_correct = False
            it.iternext()
        return is_correct


def _invdiag(data):
    result = np.zeros(data.shape)
    for index in np.ndindex(data.shape):
        if data[index] != 0:
            result[index] = 1 / data[index]
    return np.diag(result)
