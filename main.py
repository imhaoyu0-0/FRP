__author__= "Hao Yu"
__maintainer__ = "Hao Yu"
__email__ = "imhaoyu@bu.edu"
__status__ = "Developing"


import argparse
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from utils import *
parser=argparse.ArgumentParser(description='Options to train a model')
# TODO
# default setting
# Choose defect types,
parser.add_argument('--DefectType', nargs='*',help='Defect Type help')
# training data,
parser.add_argument('--datasize',
                    type=int,
                    nargs='?',
                    default=1000,
                    help='dataset help')
# choice of features
parser.add_argument('--descriptor',
                    type=str,
                    default='acsf',
                    nargs='?',help='descriptor help')
# training model
parser.add_argument('--model',
                    nargs='?',
                    choices=['lr', 'gp', 'krr','nn','rfr'],
                    help='model help')
# Optional
# ways of concatenate features
# metrics


args = parser.parse_args()

#print(args.descriptor)
#print(args.model)

# prepare dataset
address = '/projectnb/fpmats/hao/all-surface/data/data.pkl'
df = pd.read_pickle(address)
df=df.drop(df[df['Defect Number']=='1'].index)
df=df.drop(df[df['Defect Number']=='2'].index)
dataset=df
datasize=int(args.datasize)
train, test = train_test_split(dataset, train_size=datasize,test_size=datasize//5, random_state=4)
train_target = train['Surface Energy']
test_target = test['Surface Energy']

# data processing
repr_type =args.descriptor
train_embedding = []
test_embedding = []
for point in test['Object']:
    test_embedding.append(get_embedding(point, type=repr_type))
test_embedding = np.array(test_embedding)
for point in train['Object']:
    train_embedding.append(get_embedding(point, type=repr_type))
train_embedding = np.array(train_embedding)


#average predictor
result={}
avg = np.mean(train_target)
test_mse = np.sum((avg - test_target) ** 2) / len(test)
result['Average predict']= np.sqrt(test_mse)


if args.model=='lr': # linear regression
    from sklearn.linear_model import LinearRegression

    lr_reg = LinearRegression()
    lr_reg.fit(train_embedding, train_target)
    r_sq_train = lr_reg.score(train_embedding, train_target)
    r_sq_test = lr_reg.score(test_embedding, test_target)

    lr_train_pred = lr_reg.predict(train_embedding)
    lr_test_pred = lr_reg.predict(test_embedding)

    lr_test_mae = np.sum(np.abs(lr_test_pred - test_target)) / len(test_target)
    lr_test_mse = np.sum((lr_test_pred - test_target) ** 2) / len(test_target)

    lr_train_mae = np.sum(np.abs(lr_train_pred - train_target)) / len(train_target)
    lr_train_mse = np.sum((lr_train_pred - train_target) ** 2) / len(test_target)
    result['test' + repr_type + ' MAE'] = lr_test_mae
    result['train' + repr_type + ' MAE'] = lr_train_mae
    result['test' + repr_type + ' RMSE'] = np.sqrt(lr_test_mse)
    result['train' + repr_type + ' RMSE'] = np.sqrt(lr_train_mse)

    with open(repr_type +'-'+str(datasize)+ "result.json", "w") as outfile:
        json.dump(result, outfile)

elif args.model=='gp': # gaussian process
    import sklearn.gaussian_process as gp
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ExpSineSquared, ConstantKernel
    import warnings
    warnings.filterwarnings("ignore")

    kernel_options = ['dot', 'rbf', 'rbf+constant', 'dot+constant']  # 1e-7
    for KERNEL in kernel_options:
        if KERNEL == 'dot':
            kernel = DotProduct(sigma_0=1)  # +0.001*gp.kernels.Identity()
            alpha = 1e3  # 1e-5
        elif KERNEL == 'dot+constant':
            kernel = (ConstantKernel(1.0, (0.01, 1e4)) * (
                DotProduct()))  # sigma_0=0.3, sigma_0_bounds=(0.1, 10.0))**2))
            alpha = 3
        elif KERNEL == 'rbf':
            kernel = gp.kernels.RBF(1.0, (13, 1e4))
            alpha = 1e-7 * 2.7
        elif KERNEL == 'rbf+constant':
            kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e25)) * gp.kernels.RBF(10.0, (13, 1e4))
            alpha = 1e-7 * 2.7
        else:
            kernel = None
            alpha = 1e-7

        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=alpha).fit(train_embedding, train_target)

        train_score = gpr.score(train_embedding, train_target)
        test_score = gpr.score(test_embedding, test_target)

        gpr_train_pred = gpr.predict(train_embedding)
        gpr_test_pred = gpr.predict(test_embedding)

        gp_test_mae = np.sum(np.abs(gpr_test_pred - test_target)) / len(test_target)
        gpr_test_mse = np.sum((gpr_test_pred - test_target) ** 2) / len(test_target)

        gp_train_mae = np.sum(np.abs(gpr_train_pred - train_target)) / len(train_target)
        gpr_train_mse = np.sum((gpr_train_pred - train_target) ** 2) / len(train_target)

        # gpr_pred]
        print('Kernel type: %s' % KERNEL)
        print('train RMSE: %.6f' % np.sqrt(gpr_train_mse))
        print('test RMSE: %.6f' % np.sqrt(gpr_test_mse))
        print('train MAE %.6f' % gp_train_mae)
        print('test MAE %.6f' % gp_test_mae)

elif args.model=='krr': # kernel ridge regression
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.gaussian_process.kernels import ExpSineSquared
    from sklearn.model_selection import GridSearchCV

    param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
                  "kernel": [ExpSineSquared(l, p)
                             for l in np.logspace(-2, 2, 10)
                             for p in np.logspace(0, 2, 10)]}

    krr_exp = GridSearchCV(KernelRidge(), param_grid=param_grid)
    krr_exp.fit(train_embedding, train_target)
    krr_exp_pred = krr_exp.predict(test_embedding)

    krr_test_mae = np.sum(np.abs(krr_exp_pred - test_target)) / len(test_target)
    krr_exp_mse = np.sum((krr_exp_pred - test_target) ** 2) / len(test)

    krr_exp_train_pred = krr_exp.predict(train_embedding)
    krr_train_mae = np.sum(np.abs(krr_exp_train_pred - train_target)) / len(train_target)
    krr_exp_train_mse = np.sum((krr_exp_train_pred - train_target) ** 2) / len(train)
    result['test' + repr_type + ' MAE'] = krr_test_mae
    result['train' + repr_type + ' MAE'] = krr_train_mae
    result['test' + repr_type + ' RMSE'] = np.sqrt(krr_exp_mse)
    result['train' + repr_type + ' RMSE'] = np.sqrt(krr_exp_train_mse)

    with open(repr_type +'-'+str(datasize)+ "result.json", "w") as outfile:
        json.dump(result, outfile)

elif args.model=='rfr': #random forest regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV

    n_estimators = [5, 20, 50, 100, 500, 1000]
    max_features = ["auto", "sqrt", "log2"]
    max_depth = [int(x) for x in np.linspace(10, 120, num=12)]
    min_samples_split = [2, 6, 10, 20]  # minimum sample number to split a node
    min_samples_leaf = [1, 3, 4]  # minimum sample number that can be stored in a leaf node
    bootstrap = [True, False]  # method used to sample data points

    param_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }
    rfr = RandomForestRegressor()
    rfr_reg = RandomizedSearchCV(rfr, param_distributions=param_grid, n_iter=500,
                                 n_jobs=-1, cv=5, random_state=35)

    rfr_reg.fit(train_embedding, train_target)
    r_sq_train = rfr_reg.score(train_embedding, train_target)
    r_sq_test = rfr_reg.score(test_embedding, test_target)

    lr_train_pred = rfr_reg.predict(train_embedding)
    lr_test_pred = rfr_reg.predict(test_embedding)

    lr_test_mae = np.sum(np.abs(lr_test_pred - test_target)) / len(test_target)
    lr_test_mse = np.sum((lr_test_pred - test_target) ** 2) / len(test_target)

    lr_train_mae = np.sum(np.abs(lr_train_pred - train_target)) / len(train_target)
    lr_train_mse = np.sum((lr_train_pred - train_target) ** 2) / len(test_target)
    result['test' + repr_type + ' MAE'] = lr_test_mae
    result['train' + repr_type + ' MAE'] = lr_train_mae
    result['test' + repr_type + ' RMSE'] = np.sqrt(lr_test_mse)
    result['train' + repr_type + ' RMSE'] = np.sqrt(lr_train_mse)

    print('Best Parameters:', rfr_reg.best_params_)
    with open(repr_type+'-'+str(datasize) + "result.json", "w") as outfile:
        json.dump(result, outfile)
elif args.model=='nn' : # neural network, trying architectures
    raise NotImplementedError

else:
    raise NotImplementedError
