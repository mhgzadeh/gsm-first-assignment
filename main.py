from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    t = np.array([0.000, 0.375, 0.475, 1.450, 2.050])
    s = np.array([0.411, 0.330, 0.320, 0.300, 0.410])
    t0 = 0.621
    star = '*' * 15

    # ********* Q2 *********
    print('\n', '*' * 12, 'Q2', '*' * 12)
    ta = t[2]
    tb = t[3]
    sa = s[2]
    sb = s[3]
    print(f"ta: {ta},\ttb: {tb}")

    # ********* Q3 *********
    print('\n', star, 'Q3', star)
    s0 = sa + ((sb - sa) / (tb - ta)) * (t0 - ta)
    print(f"s0: {s0}")

    # ********* Q4 *********
    print('\n', star, 'Q4', star)
    a = (sb - sa) / (tb - ta)
    alpha = np.arctan(a) * (180 / np.pi)
    print(f"a: {a},\t alpha: {alpha}")

    # ********* Q5 *********
    print('\n', star, 'Q5', star)
    wa = 1 - ((t0 - ta) / (tb - ta))
    wb = (t0 - ta) / (tb - ta)
    print(f"wa: {wa},\t wb: {wb}")

    # ********* Q6 *********
    print('\n', star, 'Q6', star)
    print('Plot 1')
    plt.gcf().number
    plt.figure(1)
    plt.plot(t, s, 'ro', t, s, 'b-')
    plt.legend(['Points', 'Linear Interpolation'], loc='best')
    plt.title('Linear Interpolation')
    plt.xlabel('t')
    plt.ylabel('s')
    plt.show()

    # ********* Q7 *********
    print('\n', star, 'Q7', star)
    linear_int = interpolate.interp1d(t, s)
    s0_linear = linear_int(t0)
    print(f"s0 Linear Interpolation: {s0_linear}")

    nn_int = interpolate.interp1d(t, s, kind='nearest', fill_value="extrapolate")
    s0_nn = nn_int(t0)
    print(f"s0 nearest neighbour Interpolation: {s0_nn}")

    spline_int = interpolate.splrep(t, s)
    s0_spline = interpolate.splev(t0, spline_int)
    print(f"s0 spline Interpolation: {s0_spline}")

    # ********* Q8 *********
    print('\n', star, 'Q8', star)
    a_mat = np.empty((0, 5), int)
    for i in t:
        a_mat = np.append(a_mat, np.array([[1, i, i ** 2, i ** 3, i ** 4]]), axis=0)
    x_legendre = np.linalg.inv(a_mat).dot(s)
    print(f"a0: {x_legendre[0]},\ta1: {x_legendre[1]},\ta2: {x_legendre[2]},\t"
          f"a3: {x_legendre[3]},\ta4: {x_legendre[4]},\t")

    # ********* Q9 *********
    print('\n', star, 'Q9', star)
    s0_legendre = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4]]).dot(x_legendre)
    print(f"s0 Legendre Interpolation: {s0_legendre[0]}")

    # ********* Q10 *********
    print('\n', star, 'Q10', star)
    tt_legendre = np.linspace(-0.1, 2.2, num=47, endpoint=True)
    print(f"Grid t: {tt_legendre}")
    # plt.figure(7)
    # plt.vlines(tt_legendre, ss_legendre, 'y-')
    # plt.legend(['Grid Lines'], loc='best')
    # plt.title('Grid Spacing')
    # plt.xlabel('t')
    # plt.ylabel('s')
    # plt.show()

    # ********* Q11 *********
    print('Plot 11')
    print('\n', star, 'Q11', star)
    ss_legendre = np.array([np.array([[1, t, t ** 2, t ** 3, t ** 4]]).dot(x_legendre)[0] for t in tt_legendre])
    plt.figure(2)
    plt.plot(t, s, 'ro', tt_legendre, ss_legendre, 'g-')
    plt.legend(['Points', 'Legendre Interpolation'], loc='best')
    plt.title('Legendre Interpolation')
    plt.xlabel('t')
    plt.ylabel('s')
    plt.show()

    plt.figure(7)
    plt.vlines(tt_legendre, ss_legendre, 'g-')
    plt.legend(['Points', 'Legendre Interpolation'], loc='best')
    plt.title('Legendre Interpolation')
    plt.xlabel('t')
    plt.ylabel('s')
    plt.show()

    # ********* Q12 *********
    print('Plot 12')
    print('\n', star, 'Q12', star)

    ss_spline = interpolate.splev(tt_legendre, spline_int)
    plt.figure(3)
    plt.plot(t, s, 'ro', tt_legendre, ss_spline, 'y-')
    plt.legend(['Points', 'Spline Interpolation'], loc='best')
    plt.title('Spline Interpolation')
    plt.xlabel('t')
    plt.ylabel('s')
    plt.show()

    ss_nn = [nn_int(t) for t in tt_legendre]
    i_num = []
    for i in range(len(ss_nn) - 1):
        if ss_nn[i] != ss_nn[i + 1]:
            ss_nn[i] = None
            i_num.append(i)

    plt.figure(4)
    plt.plot(t, s, 'ro', tt_legendre, ss_nn, 'c-')
    for i in i_num:
        plt.plot(tt_legendre[(i_num[1] - 1):(i_num[1] + 1)], [ss_nn[(i_num[1] - 1)], ss_nn[(i_num[1] - 1)]], 'c-')
    plt.legend(['Points', 'Nearest-Neighbour Interpolation'], loc='best')
    plt.title('Nearest-Neighbour Interpolation')
    plt.xlabel('t')
    plt.ylabel('s')
    plt.show()

    plt.figure(5)
    plt.plot(t, s, 'ro',
             t, s, 'b-',
             tt_legendre, ss_legendre, 'g-',
             tt_legendre, ss_spline, 'y-',
             tt_legendre, ss_nn, 'c-')
    for i in i_num:
        plt.plot(tt_legendre[(i_num[1] - 1):(i_num[1] + 1)],
                 [ss_nn[(i_num[1] - 1)], ss_nn[(i_num[1] - 1)]], 'c-')
    plt.legend(['Points', 'Linear Interpolation',
                'Legendre Interpolation', 'Spline Interpolation',
                'Nearest-Neighbour Interpolation'],
               loc='best')
    plt.title('Comparison Between Methods of Interpolation')
    plt.xlabel('t')
    plt.ylabel('s')
    plt.show()

    # ********* Q13 *********
    print('\n', star, 'Q13', star)
    min_value = [abs(x - t0) for x in tt_legendre]
    nearest_num_t0_index = min_value.index(min(min_value))
    ta = tt_legendre[nearest_num_t0_index - 1]
    tb = tt_legendre[nearest_num_t0_index]
    tc = tt_legendre[nearest_num_t0_index + 1]
    td = tt_legendre[nearest_num_t0_index + 2]
    print(f"ta: {ta},\ttb: {tb},\ttc: {tc}")
    sa = np.array(np.array([[1, ta, ta ** 2, ta ** 3, ta ** 4]]).dot(x_legendre)[0])
    sb = np.array(np.array([[1, tb, tb ** 2, tb ** 3, tb ** 4]]).dot(x_legendre)[0])
    sd = np.array(np.array([[1, td, td ** 2, td ** 3, td ** 4]]).dot(x_legendre)[0])
    sc = np.array(np.array([[1, tc, tc ** 2, tc ** 3, tc ** 4]]).dot(x_legendre)[0])
    print(f"sa: {sa},\tsb: {sb},\tsc: {sc}")

    # ********* Q14 *********
    print('\n', star, 'Q14', star)
    grad_a = (sb - sa) / (tb - ta)
    grad_b = (sc - sb) / (tc - tb)
    grad_ab = (grad_b + grad_a) / 2
    print(f"grad a: {grad_a},\tgrad b: {grad_b},\tgrad ab: {grad_ab}")
    curve_math = (((sc - sb) / (tc - tb)) - ((sb - sa) / (tb - ta))) / (tb - ta)
    curve_geom = curve_math / (1 + grad_b ** 2) ** (3 / 2)
    print(f'curve math: {curve_math},\tcurve geom: {curve_geom}')

    # ********* Q15 *********
    print('\n', star, 'Q15', star)
    dev_1_t0 = np.array(np.array([[1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3]]).dot(x_legendre[1:])[0])
    dev_2_t0 = np.array(np.array([[2, 6 * t0, 12 * t0 ** 2]]).dot(x_legendre[2:])[0])
    curve_dev = dev_2_t0 / (1 + dev_1_t0 ** 2) ** (3 / 2)
    print(f'dev 1: {dev_1_t0},\tdev 2: {dev_2_t0},\tcurve dev: {curve_dev}')

    diff_grad_b_dev_1_t0 = grad_b - dev_1_t0
    diff_curve_m_dev_2_t0 = curve_math - dev_2_t0
    diff_curve_geo_curve_dev_t0 = curve_geom - curve_dev
    print(f'grad_b - dev_1: {diff_grad_b_dev_1_t0},\t'
          f'curve_m - dev_2_: {diff_curve_m_dev_2_t0},\t'
          f'curve_geo - curve_dev: {diff_curve_geo_curve_dev_t0}')

