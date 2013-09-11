#!/usr/bin/env python
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
"""
==========================================================
Prediction utility for trained EnsembleSelectionClassifier
==========================================================

Get predictions from trained EnsembleSelectionClassifier given
svm format data file.

Can output predicted classes or probabilities from the full
ensemble or just the best model.

Expects to find a trained ensemble in the sqlite db specified.

usage: ensemble_predict.py [-h] [-s {best,ens}] [-p] db_file data_file

Get EnsembleSelectionClassifier predictions

positional arguments:
  db_file        sqlite db file containing model
  data_file      testing data in svm format

optional arguments:
  -h, --help     show this help message and exit
  -s {best,ens}  choose source of prediction ["best", "ens"]
  -p             predict probabilities
"""
from __future__ import print_function

import numpy as np

from argparse import ArgumentParser

from sklearn.datasets import load_svmlight_file

from ensemble import EnsembleSelectionClassifier


def parse_args():
    desc = 'Get EnsembleSelectionClassifier predictions'
    parser = ArgumentParser(description=desc)

    parser.add_argument('db_file', help='sqlite db file containing model')
    parser.add_argument('data_file', help='testing data in svm format')

    help_fmt = 'choose source of prediction ["best", "ens"] (default "ens")'
    parser.add_argument('-s', dest='pred_src',
                        choices=['best', 'ens'],
                        help=help_fmt, default='ens')

    parser.add_argument('-p', dest='return_probs',
                        action='store_true', default=False,
                        help='predict probabilities')

    return parser.parse_args()


if (__name__ == '__main__'):
    res = parse_args()

    X, _ = load_svmlight_file(res.data_file)
    X = X.toarray()

    ens = EnsembleSelectionClassifier(db_file=res.db_file, models=None)

    if (res.pred_src == 'best'):
        preds = ens.best_model_predict_proba(X)
    else:
        preds = ens.predict_proba(X)

    if (not res.return_probs):
        preds = np.argmax(preds, axis=1)

    for p in preds:
        if (res.return_probs):
            mesg = " ".join(["%.5f" % v for v in p])
        else:
            mesg = p

        print(mesg)
