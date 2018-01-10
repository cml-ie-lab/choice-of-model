import os

import config as cf
import env_reader as er
import result_writer as rw
import sut_checker as sc
import sut_reader as sr
import transformation_model_0 as m0
import transformation_model_a as ma
import transformation_model_b as mb
import transformation_model_c as mc
import transformation_model_d as md


def main():
    """This script will read supply-use tables for a series of countries and convert them
    in four different input-output tables following transformation model A - D and model 0.
    These four different input-output tables and model 0 are subsequently used to
    calculate the total marginal output change and CO2 emissions for 1 million Euro
    final demand of each product. Subsequently scatter plots are made to show
    differences between the transformation models in terms of marginal changes in
    total sector/product output and CO2 emissions."""

    # 0. INITIALIZING
    debug = False
    save_excel = True
    configuration = cf.Config()
    if debug:
        cntr_codes = ['NL']
        cntr_names = ['Netherlands']
    else:
        cntr_codes = configuration.country_codes
        cntr_names = configuration.country_names
    yr = configuration.base_year
    data_dir = configuration.data_dir

    for cntr_code in cntr_codes:

        # 1. READING DATA
        # 1.1. status message
        cntr_name = cntr_names[cntr_codes.index(cntr_code)]
        print('start analysis for country with country code: ' + cntr_name)

        # 1.2. reading the sut file
        fn = cntr_code + '_sut_' + str(yr) + '.xlsx'
        full_fn = os.path.join(data_dir, fn)
        reader = sr.SutReader(full_fn, cntr_code, yr)
        st = reader.read_sut()

        # 1.3. reading the environmental extensions file
        env_reader = er.EnvReader(cntr_code, yr)
        e_extensions = env_reader.get_extensions()

        # 1.4. checking the sut file
        checker = sc.SutChecker(st, relative=1)
        checker.logical_checks()
        checker.value_checks()
        log = checker.log

        # 2. CREATE TRANSFORMATION MODELS
        md_0 = m0.TransformationModel0(st, e_extensions)
        md_a = ma.TransformationModelA(st, e_extensions)
        md_b = mb.TransformationModelB(st, e_extensions)
        md_c = mc.TransformationModelC(st, e_extensions)
        md_d = md.TransformationModelD(st, e_extensions)

        # 4. CREATE INPUT-OUTPUT COEFFICIENT MATRICES
        model_0 = md_0.io_matrix_model_0()
        model_a = md_a.io_coefficient_matrix()
        model_b = md_b.io_coefficient_matrix()
        model_c = md_c.io_coefficient_matrix()
        model_d = md_d.io_coefficient_matrix()

        # 5. CREATE EXTENSION COEFFICIENT MATRICES
        model_0_ext = md_0.ext_transaction_matrix()
        model_a_ext = md_a.ext_coefficients_matrix()
        model_b_ext = md_b.ext_coefficients_matrix()
        model_c_ext = md_c.ext_coefficients_matrix()
        model_d_ext = md_d.ext_coefficients_matrix()

        # 6. CHECK TRANSFORMATION MODELS
        if not md_a.check_io_transaction_matrix():
            print('Model A transaction matrix not correct')
        if not md_a.check_io_coefficients_matrix():
            print('Model A coefficients matrix not correct')
        if not md_a.check_ext_transaction_matrix():
            print('Model A extension matrix not correct')
        if not md_a.check_ext_coefficient_matrix():
            print('Model A extension coefficients matrix not correct')

        if not md_b.check_io_transaction_matrix():
            print('Model B transaction matrix not correct')
        if not md_b.check_io_coefficients_matrix():
            print('Model B coefficients matrix not correct')
        if not md_b.check_ext_transaction_matrix():
            print('Model B extension matrix not correct')
        if not md_b.check_ext_coefficient_matrix():
            print('Model B extension coefficients matrix not correct')

        if not md_c.check_io_transaction_matrix():
            print('Model C transaction matrix not correct')
        if not md_c.check_io_coefficients_matrix():
            print('Model C coefficients matrix not correct')
        if not md_c.check_ext_transaction_matrix():
            print('Model C extension matrix not correct')
        if not md_c.check_ext_coefficient_matrix():
            print('Model C extension coefficients matrix not correct')

        if not md_d.check_io_transaction_matrix():
            print('Model D transaction matrix not correct')
        if not md_d.check_io_coefficients_matrix():
            print('Model D coefficients matrix not correct')
        if not md_d.check_ext_transaction_matrix():
            print('Model D extension matrix not correct')
        if not md_d.check_ext_coefficient_matrix():
            print('Model D extension coefficients matrix not correct')

        # model 0 does not have coefficient matrices
        if not md_0.check_io_matrix():
            print('Model 0 technology matrix not correct')
        if not md_0.check_ext_matrix():
            print('Model 0 extensions matrix not correct')

        # 10. SAVE TO FILE
        if save_excel:
            fn = 'result_' + cntr_code + '_' + str(configuration.base_year) + '.xlsx'
            result_dir = configuration.results_dir
            full_fn = os.path.join(result_dir, fn)
            writer = rw.ResultWriter(full_fn, st)
            writer.set_log(log)
            writer.set_env_extensions(e_extensions)
            writer.set_model_0(model_0, model_0_ext)
            writer.set_model_a(model_a, model_a_ext)
            writer.set_model_b(model_b, model_b_ext)
            writer.set_model_c(model_c, model_c_ext)
            writer.set_model_d(model_d, model_d_ext)
            writer.write_result()


main()
