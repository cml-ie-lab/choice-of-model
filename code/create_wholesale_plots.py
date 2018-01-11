import os
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

import config as cf
import env_reader as er
import matrix_inverter as mi
import sut_reader as sr
import transformation_model_0 as m0
import transformation_model_a as ma
import transformation_model_b as mb
import transformation_model_c as mc
import transformation_model_d as md


def main():
    """This script will read supply-use tables for the Netherlands and convert them
    in four different input-output tables following transformation model A - D and model 0. These
    four different input-output tables and model 0 are subsequently used to calculate
    the marginal change and CO2 emissions for each 65 sectors/commodities related to 1 million
    Euro final demand for wholesale in the Netherlands. Subsequently scatter plots are made
    to show differences between the transformation models in terms of marginal changes in
    marginal sector/product output and CO2 emissions."""

    # 0. INITIALIZING
    show_graphics = False
    save_graphics = True
    show_chart_titles = False
    resolution_graphics = Resolution.HIGH
    configuration = cf.Config()
    cntr_code = 'NL'
    cntr_name = 'Netherlands'
    product_idx = 28  # zero based
    product_name = 'wholesale'
    yr = configuration.base_year
    data_dir = configuration.data_dir
    if resolution_graphics == Resolution.LOW:
        dpi_graphics = 50
    elif resolution_graphics == Resolution.HIGH:
        dpi_graphics = 200
    else:
        dpi_graphics = 50

    # create final demand matrix
    fd_matrix = np.zeros((configuration.product_count, 1))
    fd_matrix[product_idx, 0] = 1

    # 1. READING DATA
    # 1.1. status message
    print('start analysis for country with country code: ' + cntr_name)

    # 1.2. reading the sut file
    fn = cntr_code + '_sut_' + str(yr) + '.xlsx'
    full_fn = os.path.join(data_dir, fn)
    reader = sr.SutReader(full_fn, cntr_code, yr)
    st = reader.read_sut()

    # 1.3. reading the environmental extensions file
    env_reader = er.EnvReader(cntr_code, yr)
    e_extensions = env_reader.get_extensions()

    # 2. CREATE TRANSFORMATION MODELS
    md_0 = m0.TransformationModel0(st, e_extensions)
    md_a = ma.TransformationModelA(st, e_extensions)
    md_b = mb.TransformationModelB(st, e_extensions)
    md_c = mc.TransformationModelC(st, e_extensions)
    md_d = md.TransformationModelD(st, e_extensions)

    # 3. CREATE INPUT-OUTPUT COEFFICIENT MATRICES
    model_0 = md_0.io_matrix_model_0()
    model_a = md_a.io_coefficient_matrix()
    model_b = md_b.io_coefficient_matrix()
    model_c = md_c.io_coefficient_matrix()
    model_d = md_d.io_coefficient_matrix()

    # 4. CREATE EXTENSION COEFFICIENT MATRICES
    model_0_ext = md_0.ext_transaction_matrix()
    model_a_ext = md_a.ext_coefficients_matrix()
    model_b_ext = md_b.ext_coefficients_matrix()
    model_c_ext = md_c.ext_coefficients_matrix()
    model_d_ext = md_d.ext_coefficients_matrix()

    # 5. CHECK TRANSFORMATION MODELS
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

    # 6. CREATE FINAL DEMAND 'SHOCK' VECTORS
    # 6.1. transform final demand
    delta_fd_model_0 = md_0.final_demand(fd_matrix)
    delta_fd_model_a = md_a.final_demand(fd_matrix)
    delta_fd_model_b = md_b.final_demand(fd_matrix)
    delta_fd_model_c = md_c.final_demand(fd_matrix)
    delta_fd_model_d = md_d.final_demand(fd_matrix)

    # 7. CALCULATE CHANGES IN INDUSTRY/PRODUCT OUTPUT
    eye = np.diag(np.ones(configuration.product_count))
    delta_s_model_0 = mi.inverse(model_0) @ delta_fd_model_0
    delta_q_model_a = mi.inverse(eye - model_a) @ delta_fd_model_a
    delta_q_model_b = mi.inverse(eye - model_b) @ delta_fd_model_b
    delta_g_model_c = mi.inverse(eye - model_c) @ delta_fd_model_c
    delta_g_model_d = mi.inverse(eye - model_d) @ delta_fd_model_d

    # 8. CALCULATE CHANGES IN CO2 EMISSIONS
    # 8.1 calculate change in emissions from each product/sector
    delta_ext_model_0 = model_0_ext @ np.diag(delta_s_model_0[:, 0])
    delta_ext_model_a = model_a_ext @ np.diag(delta_q_model_a[:, 0])
    delta_ext_model_b = model_b_ext @ np.diag(delta_q_model_b[:, 0])
    delta_ext_model_c = model_c_ext @ np.diag(delta_g_model_c[:, 0])
    delta_ext_model_d = model_d_ext @ np.diag(delta_g_model_d[:, 0])

    # 8.2 take out CO2 emissions and convert from kg to metric tonne
    delta_carbon_model_0 = delta_ext_model_0[0, :]
    delta_carbon_model_a = delta_ext_model_a[0, :]
    delta_carbon_model_b = delta_ext_model_b[0, :]
    delta_carbon_model_c = delta_ext_model_c[0, :]
    delta_carbon_model_d = delta_ext_model_d[0, :]

    # 9. CREATE SCATTER PLOTS
    # 8.3.1. model a vs model b, carbon emissions
    plt.rcParams['font.size'] = 12
    plt.scatter(delta_carbon_model_a, delta_carbon_model_b)
    if show_chart_titles:
        plt.title(cntr_name + ' - $\Delta$CO$_{2}$ emissions')
    plt.xlabel('$\Delta$ CO$_{2}$$^{(A)}$ (Tonne)')
    plt.ylabel('$\Delta$ CO$_{2}$$^{(B)}$ (Tonne)')
    plt.axis('scaled')
    plt.xlim(_get_range(plt))
    plt.ylim(_get_range(plt))
    if save_graphics:
        plot_fn = cntr_code + '_carbon_' + product_name + '_model_a_b.png'
        full_plot_fn = os.path.join(configuration.results_dir, plot_fn)
        plt.savefig(full_plot_fn, dpi=dpi_graphics)
    if show_graphics:
        plt.show()
    plt.close()

    # 8.3.2 model c vs model d, carbon emissions
    plt.rcParams['font.size'] = 12
    plt.scatter(delta_carbon_model_c, delta_carbon_model_d)
    if show_chart_titles:
        plt.title(cntr_name + ' - $\Delta$CO$_{2}$ emissions')
    plt.xlabel('$\Delta$ CO$_{2}$$^{(C)}$ (Tonne)')
    plt.ylabel('$\Delta$ CO$_{2}$$^{(D)}$ (Tonne)')
    plt.axis('scaled')
    plt.xlim(_get_range(plt))
    plt.ylim(_get_range(plt))
    if save_graphics:
        plot_fn = cntr_code + '_carbon_' + product_name + '_model_c_d.png'
        full_plot_fn = os.path.join(configuration.results_dir, plot_fn)
        plt.savefig(full_plot_fn, dpi=dpi_graphics)
    if show_graphics:
        plt.show()
    plt.close()

    # 8.3.3 model a vs model c, carbon emissions
    plt.rcParams['font.size'] = 12
    plt.scatter(delta_carbon_model_a, delta_carbon_model_c)
    if show_chart_titles:
        plt.title(cntr_name + ' - $\Delta$CO$_{2}$ emissions')
    plt.xlabel('$\Delta$ CO$_{2}$$^{(A)}$ (Tonne)')
    plt.ylabel('$\Delta$ CO$_{2}$$^{(C)}$ (Tonne)')
    plt.axis('scaled')
    plt.xlim(_get_range(plt))
    plt.ylim(_get_range(plt))
    if save_graphics:
        plot_fn = cntr_code + '_carbon_' + product_name + '_model_a_c.png'
        full_plot_fn = os.path.join(configuration.results_dir, plot_fn)
        plt.savefig(full_plot_fn, dpi=dpi_graphics)
    if show_graphics:
        plt.show()
    plt.close()

    # 8.3.4 model b vs model d, carbon emissions
    plt.rcParams['font.size'] = 12
    plt.scatter(delta_carbon_model_b, delta_carbon_model_d)
    if show_chart_titles:
        plt.title(cntr_name + ' - $\Delta$CO$_{2}$ emissions')
    plt.xlabel('$\Delta$ CO$_{2}$$^{(B)}$ (Tonne)')
    plt.ylabel('$\Delta$ CO$_{2}$$^{(D)}$ (Tonne)')
    plt.axis('scaled')
    plt.xlim(_get_range(plt))
    plt.ylim(_get_range(plt))
    if save_graphics:
        plot_fn = cntr_code + '_carbon_' + product_name + '_model_b_d.png'
        full_plot_fn = os.path.join(configuration.results_dir, plot_fn)
        plt.savefig(full_plot_fn, dpi=dpi_graphics)
    if show_graphics:
        plt.show()
    plt.close()

    # 8.3.5 model 0 vs model a, carbon emissions
    plt.rcParams['font.size'] = 12
    plt.scatter(delta_carbon_model_0, delta_carbon_model_a)
    if show_chart_titles:
        plt.title(cntr_name + ' - $\Delta$CO$_{2}$ emissions')
    plt.xlabel('$\Delta$ CO$_{2}$$^{(0)}$ (Tonne)')
    plt.ylabel('$\Delta$ CO$_{2}$$^{(A)}$ (Tonne)')
    plt.axis('scaled')
    plt.xlim(_get_range(plt))
    plt.ylim(_get_range(plt))
    if save_graphics:
        plot_fn = cntr_code + '_carbon_' + product_name + '_model_0_a.png'
        full_plot_fn = os.path.join(configuration.results_dir, plot_fn)
        plt.savefig(full_plot_fn, dpi=dpi_graphics)
    if show_graphics:
        plt.show()
    plt.close()

    # 8.3.6 model 0 vs model c, carbon emissions
    plt.rcParams['font.size'] = 12
    plt.scatter(delta_carbon_model_0, delta_carbon_model_c)
    if show_chart_titles:
        plt.title(cntr_name + ' - $\Delta$CO$_{2}$ emissions')
    plt.xlabel('$\Delta$ CO$_{2}$$^{(0)}$ (Tonne)')
    plt.ylabel('$\Delta$ CO$_{2}$$^{(C)}$ (Tonne)')
    plt.axis('scaled')
    plt.xlim(_get_range(plt))
    plt.ylim(_get_range(plt))
    if save_graphics:
        plot_fn = cntr_code + '_carbon_' + product_name + '_model_0_c.png'
        full_plot_fn = os.path.join(configuration.results_dir, plot_fn)
        plt.savefig(full_plot_fn, dpi=dpi_graphics)
    if show_graphics:
        plt.show()
    plt.close()


def _get_range(plotje):
    xmin, xmax = plotje.xlim()
    ymin, ymax = plotje.ylim()
    if xmin < ymin:
        range_min = xmin
    else:
        range_min = ymin
    if xmax < ymax:
        range_max = ymax
    else:
        range_max = xmax
    return range_min, range_max


class Resolution(Enum):
    LOW = 1
    HIGH = 2


main()
