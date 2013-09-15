# -*- coding: utf8
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
"""Utility module for building model library"""

from __future__ import print_function

import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem


# generic model builder
def build_models(model_class, param_grid):
    print('Building %s models' % str(model_class).split('.')[-1][:-2])

    return [model_class(**p) for p in ParameterGrid(param_grid)]


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

    for params in ParameterGrid(param_grid):
        models.append(SVC(**params))

    return models


def build_kernPipelines(random_state=None):
    print('Building Kernel Approximation Pipelines')

    param_grid = {
        'n_components': xrange(5, 105, 5),
        'gamma': np.logspace(-6, 2, 9, base=2)
    }

    models = []

    for params in ParameterGrid(param_grid):
        nys = Nystroem(**params)
        lr = LogisticRegression()
        models.append(Pipeline([('nys', nys), ('lr', lr)]))

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

    for params in ParameterGrid(param_grid):
        km = KMeans(**params)
        lr = LogisticRegression()
        models.append(Pipeline([('km', km), ('lr', lr)]))

    return models


models_dict = {
    'svc': build_svcs,
    'sgd': build_sgdClassifiers,
    'gbc': build_gradientBoostingClassifiers,
    'dtree': build_decisionTreeClassifiers,
    'forest': build_randomForestClassifiers,
    'extra': build_extraTreesClassifiers,
    'kmp': build_kmeansPipelines,
    'kernp': build_kernPipelines,
}


def build_model_library(model_types=['dtree'], random_seed=None):
    models = []
    for m in model_types:
        models.extend(models_dict[m](random_state=random_seed))
    return models
