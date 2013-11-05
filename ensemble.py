# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
"""
The :mod:`ensemble` module implements the ensemble selection
technique of Caruana et al [1][2].

Currently supports f1, auc, rmse, accuracy and mean cross entropy scores
for hillclimbing.  Based on numpy, scipy, sklearn and sqlite.

Work in progress.

References
----------
.. [1] Caruana, et al, "Ensemble Selection from Libraries of Rich Models",
       Proceedings of the 21st International Conference on Machine Learning
       (ICML `04).
.. [2] Caruana, et al, "Getting the Most Out of Ensemble Selection",
       Proceedings of the 6th International Conference on Data Mining
       (ICDM `06).
"""
import os
import sys
import sqlite3
import numpy as np
from math import sqrt
from cPickle import loads, dumps
from collections import Counter

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer


def _f1(y, y_bin, probs):
    """return f1 score"""
    return f1_score(y, np.argmax(probs, axis=1))


def _auc(y, y_bin, probs):
    """return AUC score (for binary problems only)"""
    return roc_auc_score(y, probs[:, 1])


def _rmse(y, y_bin, probs):
    """return 1-rmse since we're maximizing the score for hillclimbing"""
    return 1.0 - sqrt(mean_squared_error(y_bin, probs))


def _accuracy(y, y_bin, probs):
    """return accuracy score"""
    return accuracy_score(y, np.argmax(probs, axis=1))


def _mxentropy(y, y_bin, probs):
    """return negative mean cross entropy since we're maximizing the score
    for hillclimbing"""

    # clip away from extremes to avoid under/overflows
    eps = 1.0e-7
    clipped = np.clip(probs, eps, 1.0 - eps)
    clipped /= clipped.sum(axis=1)[:, np.newaxis]

    return (y_bin * np.log(clipped)).sum() / y.shape[0]


def _bootstraps(n, rs):
    """return bootstrap sample indices for given n"""
    bs_inds = rs.randint(n, size=(n))
    return bs_inds, np.setdiff1d(range(n), bs_inds)


class EnsembleSelectionClassifier(BaseEstimator, ClassifierMixin):
    """Caruana-style ensemble selection [1][2]

    Parameters:
    -----------
    `db_file` : string
        Name of file for sqlite db backing store.

    `models` : list or None
        List of classifiers following sklearn fit/predict API, if None
        fitted models are loaded from the specified database.

    `n_best` : int (default: 5)
        Number of top models in initial ensemble.

    `n_folds` : int (default: 3)
        Number of internal cross-validation folds.

    `bag_fraction` : float (default: 0.25)
        Fraction of (post-pruning) models to randomly select for each bag.

    `prune_fraction` : float (default: 0.8)
        Fraction of worst models to prune before ensemble selection.

    `score_metric` : string (default: 'accuracy')
        Score metric to use when hillclimbing.  Must be one of
        'accuracy', 'xentropy', 'rmse', 'f1'.

    `epsilon` : float (default: 0.01)
        Minimum score improvement to add model to ensemble.  Ignored
        if use_epsilon is False.

    `max_models` : int (default: 50)
        Maximum number of models to include in an ensemble.

    `verbose` : boolean (default: False)
        Turn on verbose messages.

    `use_bootstrap`: boolean (default: False)
        If True, use bootstrap sample of entire dataset for fitting, and
        oob samples for hillclimbing for each internal CV fold instead
        of StratifiedKFolds

    `use_epsilon` : boolean (default: False)
        If True, candidates models are added to ensembles until the value
        of the score_metric fails to improve by the value of the epsilon
        parameter.  If False, models are added until the number of models
        in the cadidate ensemble reaches the value of the max_models
        parameter.

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to select
        candidates for each bag.

    References
    ----------
    .. [1] Caruana, et al, "Ensemble Selection from Libraries of Rich Models",
           Proceedings of the 21st International Conference on Machine Learning
           (ICML `04).
    .. [2] Caruana, et al, "Getting the Most Out of Ensemble Selection",
           Proceedings of the 6th International Conference on Data Mining
           (ICDM `06).
    """

    _metrics = {
        'f1': _f1,
        'auc': _auc,
        'rmse': _rmse,
        'accuracy': _accuracy,
        'xentropy': _mxentropy,
    }

    def __init__(self, db_file=None,
                 models=None, n_best=5, n_folds=3,
                 n_bags=20, bag_fraction=0.25,
                 prune_fraction=0.8,
                 score_metric='accuracy',
                 epsilon=0.01, max_models=50,
                 use_epsilon=False, use_bootstrap=False,
                 verbose=False, random_state=None):

        self.db_file = db_file
        self.models = models
        self.n_best = n_best
        self.n_bags = n_bags
        self.n_folds = n_folds
        self.bag_fraction = bag_fraction
        self.prune_fraction = prune_fraction
        self.score_metric = score_metric
        self.epsilon = epsilon
        self.max_models = max_models
        self.use_epsilon = use_epsilon
        self.use_bootstrap = use_bootstrap
        self.verbose = verbose
        self.random_state = random_state

        self._check_params()

        self._folds = None
        self._n_models = 0
        self._n_classes = 0
        self._metric = None
        self._ensemble = Counter()
        self._model_scores = []
        self._scored_models = []
        self._fitted_models = []

        self._init_db(models)

    def _check_params(self):
        """Parameter sanity checks"""

        if (not self.db_file):
            msg = "db_file parameter is required"
            raise ValueError(msg)

        if (self.epsilon < 0.0):
            msg = "epsilon must be >= 0.0"
            raise ValueError(msg)

        metric_names = self._metrics.keys()
        if (self.score_metric not in metric_names):
            msg = "score_metric not in %s" % metric_names
            raise ValueError(msg)

        if (self.n_best < 1):
            msg = "n_best must be >= 1"
            raise ValueError(msg)

        if (self.max_models < self.n_best):
            msg = "max_models must be >= n_best"
            raise ValueError(msg)

        if (not self.use_bootstrap):
            if (self.n_folds < 2):
                msg = "n_folds must be >= 2 for StratifiedKFolds"
                raise ValueError(msg)
        else:
            if (self.n_folds < 1):
                msg = "n_folds must be >= 1 with bootstrap"
                raise ValueError(msg)

    def _init_db(self, models):
        """Initialize database"""

        # db setup script
        _createTablesScript = """
            create table models (
                model_idx      integer UNIQUE NOT NULL,
                pickled_model  blob NOT NULL
            );

            create table fitted_models (
                model_idx      integer NOT NULL,
                fold_idx       integer NOT NULL,
                pickled_model  blob NOT NULL
            );

            create table model_scores (
                model_idx      integer UNIQUE NOT NULL,
                score          real NOT NULL,
                probs          blob NOT NULL
            );

            create table ensemble (
                model_idx      integer NOT NULL,
                weight         integer NOT NULL
            );
        """

        # barf if db file exists and we're making a new model
        if (models and os.path.exists(self.db_file)):
            raise ValueError("db_file '%s' already exists!" % self.db_file)

        db_conn = sqlite3.connect(self.db_file)
        with db_conn:
            db_conn.execute("pragma journal_mode = off")

        if (models):
            # build database
            with db_conn:
                db_conn.executescript(_createTablesScript)

            # populate model table
            insert_stmt = """insert into models (model_idx, pickled_model)
                             values (?, ?)"""
            with db_conn:
                vals = ((i, buffer(dumps(m))) for i, m in enumerate(models))
                db_conn.executemany(insert_stmt, vals)
                create_stmt = "create index models_index on models (model_idx)"
                db_conn.execute(create_stmt)

            self._n_models = len(models)

        else:
            curs = db_conn.cursor()
            curs.execute("select count(*) from models")
            self._n_models = curs.fetchone()[0]

            curs.execute("select model_idx, weight from ensemble")
            for k, v in curs.fetchall():
                self._ensemble[k] = v

            # clumsy hack to get n_classes
            curs.execute("select probs from model_scores limit 1")
            r = curs.fetchone()
            probs = loads(str(r[0]))
            self._n_classes = probs.shape[1]

        db_conn.close()

    def fit(self, X, y):
        """Perform model fitting and ensemble building"""

        self.fit_models(X, y)
        self.build_ensemble(X, y)
        return self

    def fit_models(self, X, y):
        """Perform internal cross-validation fit"""

        if (self.verbose):
            sys.stderr.write('\nfitting models:\n')

        if (self.use_bootstrap):
            n = X.shape[0]
            rs = check_random_state(self.random_state)
            self._folds = [_bootstraps(n, rs) for _ in xrange(self.n_folds)]
        else:
            self._folds = list(StratifiedKFold(y, n_folds=self.n_folds))

        select_stmt = "select pickled_model from models where model_idx = ?"
        insert_stmt = """insert into fitted_models
                             (model_idx, fold_idx, pickled_model)
                         values (?,?,?)"""

        db_conn = sqlite3.connect(self.db_file)
        curs = db_conn.cursor()

        for model_idx in xrange(self._n_models):

            curs.execute(select_stmt, [model_idx])
            pickled_model = curs.fetchone()[0]
            model = loads(str(pickled_model))

            model_folds = []

            for fold_idx, fold in enumerate(self._folds):
                train_inds, _ = fold
                model.fit(X[train_inds], y[train_inds])

                pickled_model = buffer(dumps(model))
                model_folds.append((model_idx, fold_idx, pickled_model))

            with db_conn:
                db_conn.executemany(insert_stmt, model_folds)

            if (self.verbose):
                if ((model_idx + 1) % 50 == 0):
                    sys.stderr.write('%d\n' % (model_idx + 1))
                else:
                    sys.stderr.write('.')

        if (self.verbose):
            sys.stderr.write('\n')

        with db_conn:
            stmt = """create index fitted_models_index
                      on fitted_models (model_idx, fold_idx)"""

            db_conn.execute(stmt)

        db_conn.close()

    def _score_models(self, db_conn, X, y, y_bin):
        """Get cross-validated test scores for each model"""

        self._metric = self._metrics[self.score_metric]

        if (self.verbose):
            sys.stderr.write('\nscoring models:\n')

        insert_stmt = """insert into model_scores (model_idx, score, probs)
                         values (?,?,?)"""

        select_stmt = """select pickled_model
                         from fitted_models
                         where model_idx = ? and fold_idx = ?"""

        # nuke existing scores
        with db_conn:
            stmt = """drop index if exists model_scores_index;
                      delete from model_scores;"""
            db_conn.executescript(stmt)

        curs = db_conn.cursor()

        # build probs array using the test sets for each internal CV fold
        for model_idx in xrange(self._n_models):
            probs = np.zeros((len(X), self._n_classes))

            for fold_idx, fold in enumerate(self._folds):
                _, test_inds = fold

                curs.execute(select_stmt, [model_idx, fold_idx])
                res = curs.fetchone()
                model = loads(str(res[0]))

                probs[test_inds] = model.predict_proba(X[test_inds])

            score = self._metric(y, y_bin, probs)

            # save score and probs array
            with db_conn:
                vals = (model_idx, score, buffer(dumps(probs)))
                db_conn.execute(insert_stmt, vals)

            if (self.verbose):
                if ((model_idx + 1) % 50 == 0):
                    sys.stderr.write('%d\n' % (model_idx + 1))
                else:
                    sys.stderr.write('.')

        if (self.verbose):
            sys.stderr.write('\n')

        with db_conn:
            stmt = """create index model_scores_index
                      on model_scores (model_idx)"""
            db_conn.execute(stmt)

    def _get_ensemble_score(self, db_conn, ensemble, y, y_bin):
        """Get score for model ensemble"""

        n_models = float(sum(ensemble.values()))
        ensemble_probs = np.zeros((len(y), self._n_classes))

        curs = db_conn.cursor()
        select_stmt = """select model_idx, probs
                         from model_scores
                         where model_idx in %s"""

        in_str = str(tuple(ensemble)).replace(',)', ')')
        curs.execute(select_stmt % in_str)

        for row in curs.fetchall():
            model_idx, probs = row
            probs = loads(str(probs))
            weight = ensemble[model_idx]
            ensemble_probs += probs * weight

        ensemble_probs /= n_models

        score = self._metric(y, y_bin, ensemble_probs)
        return score, ensemble_probs

    def _score_with_model(self, db_conn, y, y_bin, probs, n_models, model_idx):
        """compute ensemble score with specified model added"""

        curs = db_conn.cursor()
        select_stmt = """select probs
                         from model_scores
                         where model_idx = %d"""

        curs.execute(select_stmt % model_idx)
        row = curs.fetchone()

        n_models = float(n_models)
        new_probs = loads(str(row[0]))
        new_probs = (probs*n_models + new_probs)/(n_models + 1.0)

        score = self._metric(y, y_bin, new_probs)
        return score, new_probs

    def _ensemble_from_candidates(self, db_conn, candidates, y, y_bin):
        """Build an ensemble from a list of candidate models"""

        ensemble = Counter(candidates[:self.n_best])

        ens_score, ens_probs = self._get_ensemble_score(db_conn,
                                                        ensemble,
                                                        y, y_bin)

        ens_count = sum(ensemble.values())
        if (self.verbose):
            sys.stderr.write('%02d/%.3f ' % (ens_count, ens_score))

        cand_ensembles = []
        while(ens_count < self.max_models):
            # compute and collect scores after adding each candidate
            new_scores = []
            for new_model_idx in candidates:
                score, _ = self._score_with_model(db_conn, y, y_bin,
                                                  ens_probs, ens_count,
                                                  new_model_idx)

                new_scores.append({'score': score,
                                   'new_model_idx': new_model_idx})

            new_scores.sort(key=lambda x: x['score'], reverse=True)

            last_ens_score = ens_score
            ens_score = new_scores[0]['score']

            if (self.use_epsilon):
                # if score improvement is less than epsilon,
                # don't add the new model and stop
                score_diff = ens_score - last_ens_score
                if (score_diff < self.epsilon):
                    break

            new_model_idx = new_scores[0]['new_model_idx']
            ensemble.update({new_model_idx: 1})
            _, ens_probs = self._score_with_model(db_conn, y, y_bin,
                                                  ens_probs, ens_count,
                                                  new_model_idx)

            if (not self.use_epsilon):
                # store current ensemble to select best later
                ens_copy = Counter(ensemble)
                cand = {'ens': ens_copy, 'score': ens_score}
                cand_ensembles.append(cand)

            ens_count = sum(ensemble.values())
            if (self.verbose):
                if ((ens_count - self.n_best) % 8 == 0):
                    sys.stderr.write("\n         ")
                msg = '%02d/%.3f ' % (ens_count, ens_score)
                sys.stderr.write(msg)

        if (self.verbose):
            sys.stderr.write('\n')

        if (not self.use_epsilon and ens_count == self.max_models):
            cand_ensembles.sort(key=lambda x: x['score'], reverse=True)
            ensemble = cand_ensembles[0]['ens']

        return ensemble

    def _get_best_model(self, curs):
        """perform query for best scoring model"""

        select_stmt = """select model_idx, pickled_model
                         from models
                         where model_idx =
                               (select model_idx
                                from model_scores
                                order by score desc
                                limit 1)"""
        curs.execute(select_stmt)
        row = curs.fetchone()

        return row[0], loads(str(row[1]))

    def best_model(self):
        """Returns best model found after CV scoring"""

        db_conn = sqlite3.connect(self.db_file)
        _, model = self._get_best_model(db_conn.cursor())
        db_conn.close()
        return model

    def _print_best_results(self, curs, best_model_score):
        """Show best model and score"""

        sys.stderr.write('Best model CV score: %.5f\n' % best_model_score)

        _, best_model = self._get_best_model(curs)
        sys.stderr.write('Best model: %s\n\n' % repr(best_model))

    def build_ensemble(self, X, y, rescore=True):
        """Generate bagged ensemble"""

        self._n_classes = len(np.unique(y))

        db_conn = sqlite3.connect(self.db_file)
        curs = db_conn.cursor()

        # binarize
        if (self._n_classes > 2):
            y_bin = LabelBinarizer().fit_transform(y)
        else:
            y_bin = np.column_stack((1-y, y))

        # get CV scores for fitted models
        if (rescore):
            self._score_models(db_conn, X, y, y_bin)

        # get number of best models to take
        n_models = int(self._n_models * (1.0 - self.prune_fraction))
        bag_size = int(self.bag_fraction * n_models)
        if (self.verbose):
            sys.stderr.write('%d models left after pruning\n' % n_models)
            sys.stderr.write('leaving %d candidates per bag\n\n' % bag_size)

        # get indices and scores from DB
        select_stmt = """select model_idx, score
                         from model_scores
                         order by score desc
                         limit %d"""
        curs.execute(select_stmt % n_models)
        ranked_model_scores = [(r[0], r[1]) for r in curs.fetchall()]

        # print best performing model results
        best_model_score = ranked_model_scores[0][1]
        if (self.verbose):
            self._print_best_results(curs, best_model_score)
            sys.stderr.write("Ensemble scores for each bag (size/score):\n")

        ensembles = []

        # make bags and ensembles
        rs = check_random_state(self.random_state)
        for i in xrange(self.n_bags):
            # get bag_size elements at random
            cand_indices = rs.permutation(n_models)[:bag_size]

            # sort by rank
            candidates = [ranked_model_scores[ci][0] for ci in cand_indices]

            if (self.verbose):
                sys.stderr.write('Bag %02d): ' % (i+1))

            # build an ensemble with current candidates
            ensemble = self._ensemble_from_candidates(db_conn,
                                                      candidates,
                                                      y, y_bin)
            ensembles.append(ensemble)

        # combine ensembles from each bag
        for e in ensembles:
            self._ensemble += e

        # push to DB
        insert_stmt = "insert into ensemble(model_idx, weight) values (?, ?)"
        with db_conn:
            val_gen = ((mi, w) for mi, w in self._ensemble.most_common())
            db_conn.executemany(insert_stmt, val_gen)

        if (self.verbose):
            score, _ = self._get_ensemble_score(db_conn,
                                                self._ensemble,
                                                y, y_bin)

            fmt = "\nFinal ensemble (%d components) CV score: %.5f\n\n"

            sys.stderr.write(fmt % (sum(self._ensemble.values()), score))

        db_conn.close()

    def _model_predict_proba(self, X, model_idx=0):
        """Get probability predictions for a model given its index"""

        db_conn = sqlite3.connect(self.db_file)
        curs = db_conn.cursor()
        select_stmt = """select pickled_model
                         from fitted_models
                         where model_idx = ? and fold_idx = ?"""

        # average probs over each n_folds models
        probs = np.zeros((len(X), self._n_classes))
        for fold_idx in xrange(self.n_folds):
            curs.execute(select_stmt, [model_idx, fold_idx])

            res = curs.fetchone()
            model = loads(str(res[0]))

            probs += model.predict_proba(X)/float(self.n_folds)

        db_conn.close()

        return probs

    def best_model_predict_proba(self, X):
        """Probability estimates for all classes (ordered by class label)
        using best model"""

        db_conn = sqlite3.connect(self.db_file)
        best_model_idx, _ = self._get_best_model(db_conn.cursor())
        db_conn.close()

        return self._model_predict_proba(X, best_model_idx)

    def best_model_predict(self, X):
        """Predict class labels for samples in X using best model"""
        return np.argmax(self.best_model_predict_proba(X), axis=1)

    def predict_proba(self, X):
        """Probability estimates for all classes (ordered by class label)"""

        n_models = float(sum(self._ensemble.values()))

        probs = np.zeros((len(X), self._n_classes))

        for model_idx, weight in self._ensemble.items():
            probs += self._model_predict_proba(X, model_idx) * weight/n_models

        return probs

    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.argmax(self.predict_proba(X), axis=1)
