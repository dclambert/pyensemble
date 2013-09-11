#!/usr/bin/env python
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
"""
================================================
Training harness for EnsembleSelectionClassifier
================================================

Training harness for EnsembleSelectionClassifier object, implementing
Caruana-style ensemble selection.

The user can choose from the following candidate models:

    sgd     : Stochastic Gradient Descent
    svc     : Support Vector Machines
    gbc     : Gradient Boosting Classifiers
    dtree   : Decision Trees
    forest  : Random Forests
    extra   : Extra Trees
    kmp     : KMeans->LogisticRegression Pipelines
    kern    : Nystroem->LogisticRegression Pipelines

usage: ensemble_train.py [-h]
                         [-M {svc,sgd,gbc,dtree,forest,extra,kmp,kernp}
                            [{svc,sgd,gbc,dtree,forest,extra,kmp,kernp} ...]]
                         [-S {f1,auc,rmse,accuracy,xentropy}] [-b N_BAGS]
                         [-f BAG_FRACTION] [-B N_BEST] [-m MAX_MODELS]
                         [-F N_FOLDS] [-p PRUNE_FRACTION] [-u] [-U]
                         [-e EPSILON] [-t TEST_SIZE] [-s SEED] [-v]
                         db_file data_file

EnsembleSelectionClassifier training harness

positional arguments:
  db_file               sqlite db file for backing store
  data_file             training data in svm format

optional arguments:
  -h, --help            show this help message and exit
  -M {svc,sgd,gbc,dtree,forest,extra,kmp,kernp}
    [{svc,sgd,gbc,dtree,forest,extra,kmp,kernp} ...]
                        model types to include as ensemble candidates
                        (default: ['dtree'])
  -S {f1,auc,rmse,accuracy,xentropy}
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
  -u                    use epsilon to stop adding models (default: False)
  -U                    use bootstrap sample to generate training/hillclimbing
                        folds (default: False)
  -e EPSILON            score improvement threshold to include new model
                        (default: 0.0001)
  -t TEST_SIZE          fraction of data to use for testing (default: 0.75)
  -s SEED               random seed
  -v                    show progress messages
"""

from __future__ import print_function

from argparse import ArgumentParser

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split

from ensemble import EnsembleSelectionClassifier
from model_library import build_model_library


def parse_args():
    desc = 'EnsembleSelectionClassifier training harness'
    parser = ArgumentParser(description=desc)

    dflt_fmt = '(default: %(default)s)'

    parser.add_argument('db_file', help='sqlite db file for backing store')
    parser.add_argument('data_file', help='training data in svm format')

    model_choices = ['svc', 'sgd', 'gbc', 'dtree',
                     'forest', 'extra', 'kmp', 'kernp']
    help_fmt = 'model types to include as ensemble candidates %s' % dflt_fmt
    parser.add_argument('-M', dest='model_types', nargs='+',
                        choices=model_choices,
                        help=help_fmt, default=['dtree'])

    help_fmt = 'scoring metric used for hillclimbing %s' % dflt_fmt
    parser.add_argument('-S', dest='score_metric',
                        choices=['f1', 'auc', 'rmse', 'accuracy', 'xentropy'],
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

    help_fmt = ('use bootstrap sample to generate '
                'training/hillclimbing folds %s' % dflt_fmt)
    parser.add_argument('-U', dest='use_bootstrap', action='store_true',
                        help=help_fmt, default=False)

    help_fmt = 'score improvement threshold to include new model %s' % dflt_fmt
    parser.add_argument('-e', dest='epsilon', type=float,
                        help=help_fmt, default=0.0001)

    help_fmt = 'fraction of data to use for testing %s' % dflt_fmt
    parser.add_argument('-t', dest='test_size', type=float, help=help_fmt,
                        default=0.75)

    help_fmt = 'random seed'
    parser.add_argument('-s', dest='seed', type=int, help=help_fmt)

    parser.add_argument('-v', dest='verbose', action='store_true',
                        help='show progress messages', default=False)

    return parser.parse_args()


if (__name__ == '__main__'):
    res = parse_args()

    X_train, y_train = load_svmlight_file(res.data_file)
    X_train = X_train.toarray()

    # train_test_split for testing set if test_size>0.0
    if (res.test_size > 0.0):
        do_test = True
        splits = train_test_split(X_train, y_train,
                                  test_size=res.test_size,
                                  random_state=res.seed)

        X_train, X_test, y_train, y_test = splits
        print('Train/hillclimbing set size: %d' % len(X_train))
        print('              Test set size: %d\n' % len(X_test))
    else:
        do_test = False
        print('Train/hillclimbing set size: %d' % len(X_train))

    # get model lib
    models = build_model_library(res.model_types, res.seed)
    print('built %d models\n' % len(models))

    param_dict = {
        'models': models,
        'db_file': res.db_file,
        'n_best': res.n_best,
        'n_folds': res.n_folds,
        'n_bags': res.n_bags,
        'bag_fraction': res.bag_fraction,
        'prune_fraction': res.prune_fraction,
        'score_metric': res.score_metric,
        'verbose': res.verbose,
        'epsilon': res.epsilon,
        'use_epsilon': res.use_epsilon,
        'use_bootstrap': res.use_bootstrap,
        'max_models': res.max_models,
        'random_state': res.seed,
    }

    try:
        ens = EnsembleSelectionClassifier(**param_dict)
    except ValueError as e:
        print('ERROR: %s' % e)
        import sys
        sys.exit(1)

    print('fitting ensemble:\n%s\n' % ens)

    # fit models, score, build ensemble
    ens.fit(X_train, y_train)

    preds = ens.best_model_predict(X_train)
    score = accuracy_score(y_train, preds)
    print('Train set accuracy from best model: %.5f' % score)

    preds = ens.predict(X_train)
    score = accuracy_score(y_train, preds)
    print('Train set accuracy from final ensemble: %.5f' % score)

    if (do_test):
        preds = ens.best_model_predict(X_test)
        score = accuracy_score(y_test, preds)
        print('\n Test set accuracy from best model: %.5f' % score)

        fmt = '\n Test set classification report for best model:\n%s'
        report = classification_report(y_test, preds)
        print(fmt % report)

        preds = ens.predict(X_test)
        score = accuracy_score(y_test, preds)
        print(' Test set accuracy from final ensemble: %.5f' % score)

        fmt = '\n Test set classification report for final ensemble:\n%s'
        report = classification_report(y_test, preds)
        print(fmt % report)
