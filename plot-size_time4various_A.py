"""Experiment and plot figure 3 of the paper.
      Size/Time for various |A|,  m=3, d=7
"""

import numpy as np
import Booster
from matplotlib import pyplot as plt

def main():
    data_range = 100
    d = 7
    #num_of_alphas = 100
    folds = 3

    # lassoCV
    solver = "lasso"
    rec_n = list(range(100000, 2000000, 100000))
    rec_lasso_AEq50 = [0]        # [[n1,time1],[n2,time2],...]
    rec_lasso_AEq100 = [0]
    rec_lasso_AEq200 = [0]
    rec_lasso_AEq300 = [0]
    rec_lasso_boost_AEq50 = [0]  # [[n1,time1],[n2,time2],...]
    rec_lasso_boost_AEq100 = [0]
    rec_lasso_boost_AEq200 = [0]
    rec_lasso_boost_AEq300 = [0]

    for n in rec_n:
        print('calculating n={}...'.format(n))

        for num_of_alphas in [50, 100, 200, 300]:
            # generate data
            data = np.floor(np.random.rand(n, d) * data_range)
            labels = np.floor(np.random.rand(n, 1) * data_range)
            weights = np.ones(n)
    
            clf = Booster.get_new_clf(solver, folds=folds, alphas=num_of_alphas)
            time_coreset, _ = Booster.coreset_train_model(data, labels, weights, clf, folds=folds, solver=solver)
            clf = Booster.get_new_clf(solver, folds=folds, alphas=num_of_alphas)
            time_real, _ = Booster.train_model(data, labels, weights, clf)
            
            # record
            if num_of_alphas == 50:
                rec_lasso_AEq50.      append(time_real)
                rec_lasso_boost_AEq50.append(time_coreset)
            elif num_of_alphas == 100:
                rec_lasso_AEq100.      append(time_real)
                rec_lasso_boost_AEq100.append(time_coreset)
            elif num_of_alphas == 200:
                rec_lasso_AEq200.      append(time_real)
                rec_lasso_boost_AEq200.append(time_coreset)
            else:
                rec_lasso_AEq300.      append(time_real)
                rec_lasso_boost_AEq300.append(time_coreset)
            
            print('  num_of_alphas={} done'.format(num_of_alphas))
    # plot ----
    rec_n.insert(0, 0)
    fig_lasso = plt.figure()
    ax = fig_lasso.add_subplot(1, 1, 1)
    ax.set_title('size-time of various |A| - lassoCV')
    ax.set_xlabel('Data size n')
    ax.set_ylabel('time/s')
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')
    ax.plot(rec_n, rec_lasso_AEq50, 'bo-',
            rec_n, rec_lasso_AEq100, 'go-',
            rec_n, rec_lasso_AEq200, 'ro-',
            rec_n, rec_lasso_AEq300, 'mo-',
            rec_n, rec_lasso_boost_AEq50, 'b--',
            rec_n, rec_lasso_boost_AEq100, 'g--',
            rec_n, rec_lasso_boost_AEq200, 'r--',
            rec_n, rec_lasso_boost_AEq300, 'm--')

    ax.legend(['lassoCV, |A|=50', 'lassoCV, |A|=100', 'lassoCV, |A|=200','lassoCV, |A|=300',
               'lassoCV-BOOST, |A|=50', 'lassoCV-BOOST, |A|=100', 'lassoCV-BOOST, |A|=200','lassoCV-BOOST, |A|=300'])

    plt.savefig('A_lasso.png')
    plt.show()



if __name__ == '__main__':
    main()
