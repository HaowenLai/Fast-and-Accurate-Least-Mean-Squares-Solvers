"""Experiment and plot figure 2 of the paper.
      Size/Time for various d,  m=3, |A|=100
"""

import numpy as np
import Booster
from matplotlib import pyplot as plt

def main():
    data_range = 100
    num_of_alphas = 100
    folds = 3

    # ridgeCV
    solver = "ridge"
    rec_n = list(range(100000, 500000, 100000))
    rec_ridge_dEq3 = [0]        # [[n1,time1],[n2,time2],...]
    rec_ridge_dEq5 = [0]
    rec_ridge_dEq7 = [0]
    rec_ridge_boost_dEq3 = [0]  # [[n1,time1],[n2,time2],...]
    rec_ridge_boost_dEq5 = [0]
    rec_ridge_boost_dEq7 = [0]
    for n in rec_n:
        print('calculating n={}...'.format(n))

        for d in [3, 5, 7]:
            # generate data
            data = np.floor(np.random.rand(n, d) * data_range)
            labels = np.floor(np.random.rand(n, 1) * data_range)
            weights = np.ones(n)
    
            clf = Booster.get_new_clf(solver, folds=folds, alphas=num_of_alphas)
            time_coreset, _ = Booster.coreset_train_model(data, labels, weights, clf, folds=folds, solver=solver)
            clf = Booster.get_new_clf(solver, folds=folds, alphas=num_of_alphas)
            time_real, _ = Booster.train_model(data, labels, weights, clf)
            
            # record
            if d == 3:
                rec_ridge_dEq3.      append(time_real)
                rec_ridge_boost_dEq3.append(time_coreset)
            elif d == 5:
                rec_ridge_dEq5.      append(time_real)
                rec_ridge_boost_dEq5.append(time_coreset)
            else:
                rec_ridge_dEq7.      append(time_real)
                rec_ridge_boost_dEq7.append(time_coreset)
            
            print('  d={} done'.format(d))
    # plot ----
    rec_n.insert(0, 0)
    fig_ridge = plt.figure()
    ax = fig_ridge.add_subplot(1, 1, 1)
    ax.set_title('size-time of various d - ridgeCV')
    ax.set_xlabel('Data size n')
    ax.set_ylabel('time/s')
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.plot(rec_n, rec_ridge_dEq3, 'bo-',
            rec_n, rec_ridge_dEq5, 'go-',
            rec_n, rec_ridge_dEq7, 'ro-',
            rec_n, rec_ridge_boost_dEq3, 'b--',
            rec_n, rec_ridge_boost_dEq5, 'g--',
            rec_n, rec_ridge_boost_dEq7, 'r--')
    ax.legend(['ridgeCV, d=3', 'ridgeCV, d=5', 'ridgeCV, d=7',
               'ridgeCV-BOOST, d=3', 'ridgeCV-BOOST, d=5', 'ridgeCV-BOOST, d=7'])
    plt.show()


if __name__ == '__main__':
    main()
