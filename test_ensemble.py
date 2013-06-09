# -*- coding: utf8
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
"""
=============================
Tester for EnsembleClassifier
=============================

Test harness for EnsembleClassifier object, implementing
Caruana-style ensemble selection.

This uses the Letters dataset, making it a binary classification
problem, each class representing one half of the alphabet.

The user can choose from the following candidate models:

    sgd     : Stochastic Gradient Descent
    svc     : Support Vector Machines
    gbc     : Gradient Boosting Classifiers
    dtree   : Decision Trees
    forest  : Random Forests
    extra   : Extra Trees
    kmp     : KMeans->LogisticRegression Pipelines

usage: test_ensemble.py [-h] -D DB_NAME
                        [-M {svc,sgd,gbc,dtree,forest,extra,kmp}
                                   [{svc,sgd,gbc,dtree,forest,extra,kmp} ...]]
                        [-S {accuracy,rmse,xentropy,f1}] [-b N_BAGS]
                        [-f BAG_FRACTION] [-B N_BEST] [-m MAX_MODELS]
                        [-F N_FOLDS] [-p PRUNE_FRACTION] [-e EPSILON]
                        [-t TEST_SIZE] [-s SEED] [-v]

Test EnsembleClassifier

optional arguments:
  -h, --help            show this help message and exit
  -D DB_NAME            file for backing store
  -M {svc,sgd,gbc,dtree,forest,extra,kmp}
                                    [{svc,sgd,gbc,dtree,forest,extra,kmp} ...]
                        model types to include as ensemble candidates
                        (default: ['dtree'])
  -S {accuracy,rmse,xentropy,f1}
                        scoring metric used for hillclimbing (default:
                        accuracy)
  -b N_BAGS             bags to create (default: 20)
  -f BAG_FRACTION       fraction of models in each bag (after pruning)
                        (default: 0.25)
  -B N_BEST             number of best models in initial ensemble (default: 5)
  -m MAX_MODELS         maximum number of models per bagged ensemble (default:
                        25)
  -F N_FOLDS            internal cross-validation folds (default: 3)
  -p PRUNE_FRACTION     fraction of worst models pruned pre-selection
                        (default: 0.75)
  -e EPSILON            score improvement threshold to include new model
                        (default: 0.0001)
  -t TEST_SIZE          fraction of data to use for testing (default: 0.95)
  -s SEED               random seed
  -v                    show progress messages
"""

from __future__ import print_function

import numpy as np
from argparse import ArgumentParser

from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata
from sklearn.grid_search import IterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from ensemble import EnsembleSelectionClassifier


# generic model builder
def build_models(model_class, param_grid):
    print('Building %s models' % str(model_class).split('.')[-1][:-2])

    models = []
    for params in IterGrid(param_grid):
        models.append(model_class(**params))

    return models


def build_randomForestClassifiers(random_state=None):
    param_grid = {
        'n_estimators': [20, 50, 100],
        'criterion':  ['gini', 'entropy'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [1, 2, 5, 10],
        'min_density': [0.25, 0.5, 0.75, 1.0],
        'random_state': [random_state],
    }

    return build_models(RandomForestClassifier, param_grid)


def build_gradientBoostingClassifiers(random_state=None):
    param_grid = {
        'max_depth': [1, 2, 5, 10],
        'n_estimators': [10, 20, 50, 100],
        'subsample': np.linspace(0.2, 1.0, 5),
        'max_features': np.linspace(0.2, 1.0, 5),
    }

    return build_models(GradientBoostingClassifier, param_grid)


def build_sgdClassifiers(random_state=None):
    param_grid = {
        'loss': ['log', 'modified_huber'],
        'penalty': ['elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'optimal'],
        'n_iter': [2, 5, 10],
        'eta0': [0.001, 0.01, 0.1],
        'l1_ratio': np.linspace(0.0, 1.0, 3),
    }

    return build_models(SGDClassifier, param_grid)


def build_decisionTreeClassifiers(random_state=None):
    rs = check_random_state(random_state)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [None, 1, 2, 5, 10],
        'min_samples_split': [1, 2, 5, 10],
        'random_state': [rs.random_integers(100000) for i in xrange(3)],
    }

    return build_models(DecisionTreeClassifier, param_grid)


def build_extraTreesClassifiers(random_state=None):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [5, 10, 20],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'max_depth': [None, 1, 2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'random_state': [random_state],
    }

    return build_models(ExtraTreesClassifier, param_grid)


def build_svcs(random_state=None):
    print('Building SVM models')

    Cs = np.logspace(-7, 2, 10)
    gammas = np.logspace(-6, 2, 9, base=2)
    coef0s = [-1.0, 0.0, 1.0]

    models = []

    for C in Cs:
        models.append(SVC(kernel='linear', C=C, probability=True,
                          cache_size=1000))

    for C in Cs:
        for coef0 in coef0s:
            models.append(SVC(kernel='sigmoid', C=C, coef0=coef0,
                              probability=True, cache_size=1000))

    for C in Cs:
        for gamma in gammas:
            models.append(SVC(kernel='rbf', C=C, gamma=gamma,
                              cache_size=1000, probability=True))

    param_grid = {
        'kernel': ['poly'],
        'C': Cs,
        'gamma': gammas,
        'degree': [2],
        'coef0': coef0s,
        'probability': [True],
        'cache_size': [1000],
    }

    for params in IterGrid(param_grid):
        models.append(SVC(**params))

    return models


def build_kmeansPipelines(random_state=None):
    print('Building KMeans-Logistic Regression Pipelines')

    param_grid = {
        'n_clusters': xrange(5, 205, 5),
        'init': ['k-means++', 'random'],
        'n_init': [1, 2, 5, 10],
        'random_state': [random_state],
    }

    models = []

    for params in IterGrid(param_grid):
        km = KMeans(**params)
        lr = LogisticRegression()
        models.append(Pipeline([('km', km), ('lr', lr)]))

    return models


def load_data(test_size=0.25, random_state=None):
    print('\nloading letter data')

    letter = fetch_mldata('letter')
    X = letter.data
    y = np.array(letter.target > 12, dtype=int)

    splits = train_test_split(X, y, test_size=test_size,
                              random_state=random_state)
    X_train, X_test, y_train, y_test = splits

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def parse_args():
    parser = ArgumentParser(description='Test EnsembleClassifier')

    dflt_fmt = '(default: %(default)s)'

    parser.add_argument('-D', dest='db_name', required=True,
                        help='file for backing store')

    model_choices = ['svc', 'sgd', 'gbc', 'dtree', 'forest', 'extra', 'kmp']
    help_fmt = 'model types to include as ensemble candidates %s' % dflt_fmt
    parser.add_argument('-M', dest='model_types', nargs='+',
                        choices=model_choices,
                        help=help_fmt, default=['dtree'])

    help_fmt = 'scoring metric used for hillclimbing %s' % dflt_fmt
    parser.add_argument('-S', dest='score_metric',
                        choices=['accuracy', 'rmse', 'xentropy', 'f1'],
                        help=help_fmt,  default='accuracy')

    parser.add_argument('-b', dest='n_bags', type=int,
                        help='bags to create (default: %(default)s)',
                        default=20)

    help_fmt = 'fraction of models in each bag (after pruning) %s' % dflt_fmt
    parser.add_argument('-f', dest='bag_fraction', type=float,
                        help=help_fmt, default=.25)

    help_fmt = 'number of best models in initial ensemble %s' % dflt_fmt
    parser.add_argument('-B', dest='n_best', type=int,
                        help=help_fmt, default=5)

    help_fmt = 'maximum number of models per bagged ensemble %s' % dflt_fmt
    parser.add_argument('-m', dest='max_models', type=int,
                        help=help_fmt, default=25)

    help_fmt = 'internal cross-validation folds %s' % dflt_fmt
    parser.add_argument('-F', dest='n_folds', type=int,
                        help=help_fmt, default=3)

    help_fmt = 'fraction of worst models pruned pre-selection %s' % dflt_fmt
    parser.add_argument('-p', dest='prune_fraction', type=float,
                        help=help_fmt, default=0.75)

    help_fmt = 'use epsilon to stop adding models %s' % dflt_fmt
    parser.add_argument('-u', dest='use_epsilon', action='store_true',
                        help=help_fmt, default=False)

    help_fmt = 'score improvement threshold to include new model %s' % dflt_fmt
    parser.add_argument('-e', dest='epsilon', type=float,
                        help=help_fmt, default=0.0001)

    help_fmt = 'fraction of data to use for testing %s' % dflt_fmt
    parser.add_argument('-t', dest='test_size', type=float, help=help_fmt,
                        default=0.95)

    help_fmt = 'random seed'
    parser.add_argument('-s', dest='seed', type=int, help=help_fmt)

    parser.add_argument('-v', dest='verbose', action='store_true',
                        help='show progress messages', default=True)

    return parser.parse_args()


def main():
    res = parse_args()

    data = load_data(res.test_size, random_state=res.seed)
    X_train, X_test, y_train, y_test = data

    print('Train/hillclimbing set size: %d' % len(X_train))
    print('              Test set size: %d\n' % len(X_test))

    models_dict = {
        'svc': build_svcs,
        'sgd': build_sgdClassifiers,
        'gbc': build_gradientBoostingClassifiers,
        'dtree': build_decisionTreeClassifiers,
        'forest': build_randomForestClassifiers,
        'extra': build_extraTreesClassifiers,
        'kmp': build_kmeansPipelines,
    }

    models = []
    for m in res.model_types:
        models.extend(models_dict[m](random_state=res.seed))

    print('built %d models\n' % len(models))

    param_dict = {
        'models': models,
        'db_name': res.db_name,
        'n_best': res.n_best,
        'n_folds': res.n_folds,
        'n_bags': res.n_bags,
        'bag_fraction': res.bag_fraction,
        'prune_fraction': res.prune_fraction,
        'score_metric': res.score_metric,
        'verbose': res.verbose,
        'epsilon': res.epsilon,
        'use_epsilon': res.use_epsilon,
        'max_models': res.max_models,
        'random_state': res.seed,
    }

    ens = EnsembleSelectionClassifier(**param_dict)

    print('fitting ensemble:\n%s\n' % ens)

    ens.fit(X_train, y_train)

    preds = ens.best_model_predict(X_train)
    score = accuracy_score(y_train, preds)
    print('Train set accuracy from best model: %.5f' % score)

    preds = ens.best_model_predict(X_test)
    score = accuracy_score(y_test, preds)
    print(' Test set accuracy from best model: %.5f' % score)

    report = classification_report(y_test, preds)
    print('\n Test set classification report for best model:\n%s' % report)

    preds = ens.predict(X_train)
    score = accuracy_score(y_train, preds)
    print('\nTrain set score from final ensemble: %.5f' % score)

    preds = ens.predict(X_test)
    score = accuracy_score(y_test, preds)
    print(' Test set score from final ensemble: %.5f' % score)

    report = classification_report(y_test, preds)
    print('\n Test set classification report for final ensemble:\n%s' % report)

if (__name__ == '__main__'):
    main()
