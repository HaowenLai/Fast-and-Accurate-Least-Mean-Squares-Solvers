"""Experiment and plot figure 10 of the paper.
Using data set (i), 3D Road Network (North Jutland, Denmark)
Accuracy comparison between:
    linreg + boost;
    sketch + inverse;
    sketch + cholesky
Plot line chart and histogam.
"""

import numpy as np
import Booster
from matplotlib import pyplot as plt
from matplotlib import ticker

def load_data(path):
    """Load raw data from `path` and return data and labels.
    Param:
        path: str, path to the raw data file.
    Return:
        A_b: (A|b) in ||Ax-b||
    Raise:
        OSError: file NOT found.
    """
    d = np.loadtxt(path, np.float32, delimiter = ',', skiprows = 1)
    return d[:, 1:]


def sketch_inverse(A, b):
    """Implement of `sketch+inverse' mentioned in the paper.
    Param:
        A, b: n*d, n*1 np.ndarray, coefficient in ||Ax-b||
    Return:
        x_inverse: d*1 np.ndarray, optimal x of this method.
    """
    x_inverse = np.linalg.inv(A.T @ A) @ A.T @ b
    return x_inverse


def sketch_cholesky(A, b):
    """Implement of `sketch+cholesky' mentioned in the paper.
    Param:
        A, b: n*d, n*1 np.ndarray, coefficient in ||Ax-b||
    Return:
        x_cholesky: d*1 np.ndarray, optimal x of this method.
    """
    A1 = np.hstack((A, b))
    S = np.linalg.cholesky(A1.T @ A1).T  # A1.T * A1 = S.T*S
    # //
    data = S[:,:2]
    labels = S[:, 2:]
    weights = np.ones(len(labels))

    # standard method
    clf = Booster.get_new_clf('linear')
    _, clf = Booster.train_model(data, labels, weights, clf)
    x_cholesky = clf.coef_.astype(np.float32)
    
    return x_cholesky


def error_func(A, b, x, x_star, fold=50):
    """Calculate errors ||Ax_1-b||-||Ax_star-b||, where x1 \in x.
    Param:
        A: n*d np.ndarray, coefficient in ||Ax-b||
        b: n*1 np.ndarray, coefficient in ||Ax-b||
        x: tuple, (x_linBoost, x_inverse, x_cholesky)
        x_star: d*1 np.ndarray, x* by lstsq()
        fold: if MomeryError occurs, increase this parameter.
    Return:
        error: tuple, (e_linBoost, e_inverse, e_cholesky)
    """
    factor = len(b) // fold + 1
    
    # ||Ax_star-b||_2^2
    sum2_star = np.float64(.0)
    for i in range(fold):
        sum2_star += np.linalg.norm(
            A[i*factor:(i + 1)*factor] @ x_star - b[i*factor:(i + 1)*factor])**2
    
    # ||Ax-b||_2^2
    x_linBoost, x_inverse, x_cholesky = x
    sum2_linBoost = np.float32(.0)
    sum2_inverse  = np.float32(.0)
    sum2_cholesky = np.float32(.0)
    # //
    for i in range(fold):
        A1 = A[i * factor:(i + 1) * factor]
        b1 = b[i * factor:(i + 1) * factor]
        sum2_linBoost += np.linalg.norm(A1 @ x_linBoost - b1)** 2
        sum2_inverse  += np.linalg.norm(A1 @ x_inverse  - b1)** 2
        sum2_cholesky += np.linalg.norm(A1 @ x_cholesky - b1)** 2

    # result error
    error = np.sqrt(sum2_linBoost) - np.sqrt(sum2_star), \
            np.sqrt(sum2_inverse ) - np.sqrt(sum2_star), \
            np.sqrt(sum2_cholesky) - np.sqrt(sum2_star)
    return error


def main():
    # load data
    path = './data/3D_spatial_network.csv'
    A_b = load_data(path)  # (A|b) or (data|labels)
    
    solver = 'linear'

    # records
    rec_n = np.arange(5000, 430001, 5000)
    rec_err = np.empty((430001 // 5000, 3), dtype=np.float32)

    for i, n in enumerate(rec_n):
        # generate n size A and b
        np.random.shuffle(A_b)
        A = A_b[:n,:2]
        b = A_b[:n, 2:]
        weights = np.ones(n)

        # standard method
        clf = Booster.get_new_clf(solver)
        _, clf_star = Booster.train_model(A, b, weights, clf)
        x_star = clf_star.coef_.astype(np.float32)
        x_star = x_star[:, np.newaxis]  # convert to shape(d,1)
        
        # LINREG+BOOST
        clf = Booster.get_new_clf(solver)
        _, clf_linBoost = Booster.coreset_train_model(A, b, weights, clf, solver=solver)
        x_linBoost = clf_linBoost.coef_.astype(np.float32)
        x_linBoost = x_linBoost[:, np.newaxis]  # convert to shape(d,1)
        
        # SKETCH+INVERSE
        x_inverse = sketch_inverse(A, b) # don not need convert
        
        # SKETCH+CHOLESKY
        x_cholesky = sketch_cholesky(A, b)
        x_cholesky = x_cholesky[:, np.newaxis]  # convert to shape(d,1)

        # compute the errors
        e = error_func(A, b, (x_linBoost, x_inverse, x_cholesky), x_star)
        rec_err[i] = e
    
    rec_err = np.abs(rec_err)
    
    # plot line chart------
    formatter = ticker.ScalarFormatter(useMathText=True) # formatter
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    fig_lineChart = plt.figure()
    ax = fig_lineChart.add_subplot(1, 1, 1)
    ax.set_title('line chart of errors of three methods')
    ax.set_xlabel('Data Size n')
    ax.set_ylabel('$||Ax-b||_2$-$||Ax^*-b||_2$')
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.yaxis.set_major_formatter(formatter)
    ax.plot(rec_n, rec_err[:,0], 'r^-',
            rec_n, rec_err[:,1], 'b^-',
            rec_n, rec_err[:,2], 'g^-',)
    ax.legend(['LINREG+BOOST', 'SKETCH+INVERSE', 'SKETCH+CHOLESKY'])
    # //
    # plot histogam
    upper_bound = np.max(rec_err)
    fig_hist = plt.figure()
    ax = fig_hist.add_subplot(1, 1, 1)
    ax.set_title('histogam of errors of three methods')
    ax.set_xlabel('$||Ax-b||_2$-$||Ax^*-b||_2$')
    ax.set_ylabel('Count')
    ax.xaxis.grid(False, which='major')
    ax.yaxis.grid(True, which='major')
    ax.hist(rec_err[:, 0], bins=40, range=(0, upper_bound), color="r")
    ax.hist(rec_err[:, 1], bins=40, range=(0, upper_bound), color="b")
    ax.hist(rec_err[:, 2], bins=40, range=(0, upper_bound), color="g")
    ax.xaxis.set_major_formatter(formatter)
    ax.legend(['LINREG+BOOST', 'SKETCH+INVERSE', 'SKETCH+CHOLESKY'])
    # plot small range histogam
    fig_hist1 = plt.figure()
    ax = fig_hist1.add_subplot(1, 1, 1)
    ax.set_title('small range histogam of errors of LINREG+BOOST')
    ax.set_xlabel('$||Ax-b||_2$-$||Ax^*-b||_2$')
    ax.set_ylabel('Count')
    ax.xaxis.grid(False, which='major')
    ax.yaxis.grid(True, which='major')
    ax.hist(rec_err[:, 0], bins=12, range=(0, upper_bound/4), color="r")
    ax.xaxis.set_major_formatter(formatter)
    ax.legend(['LINREG+BOOST'])
    # //
    plt.show()


if __name__ == '__main__':
    main()
