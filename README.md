pyensemble v0.41
================

###### An implementation of [Caruana et al's Ensemble Selection algorithm] (http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf) [1][2] in Python, based on [scikit-learn](http://scikit-learn.org).

###### From the abstract:

> We present a method for constructing ensembles from libraries of thousands of models.
Model libraries are generated using different learning algorithms and parameter settings.
Forward stepwise selection is used to add to the ensemble the models that maximize its
performance.  Ensemble selection allows ensembles to be optimized to performance metrics
such as accuracy, cross entropy, mean precision or ROC Area.  Experiments with seven test
problems and ten metrics demonstrate the benefit of ensemble selection.

It's a work in progress, so things can/might/will change.

__David C. Lambert__  
__dcl [at] panix [dot] com__  

__Copyright © 2013__  
__License: Simple BSD__

Files
-----

#### __ensemble.py__

Containing the EnsembleSelectionClassifier object

The EnsembleSelectionClassifier object tries to implement all of the methods in the combined
paper, including internal cross validation, bagged ensembling, initialization with the best
models, pruning of the worst models prior to selection, and sampling with replacement of the
model candidates.

It uses sqlite as the backing store containing pickled unfitted models, fitted model 'siblings'
for each internal cross validation fold, scores and predictions for each model, and the list of
model ids and weightings for the final ensemble.

Hillclimbing can be performed using auc, accuracy, rmse, cross entropy or F1 score.

If the object is initialized with the _model_ parameter equal to None, the object tries to load
a fitted ensemble from the database specified.

__*(NOTE: Expects class labels to be sequential integers starting at zero [for now].)*__
    
#### __model_library.py__

Example model library building code.

#### __ensemble_train.py__

Training utility to run ensemble selection on svm data files.

The user can choose from the following candidate models:

*    sgd     : Stochastic Gradient Descent
*    svc     : Support Vector Machines
*    gbc     : Gradient Boosting Classifiers
*    dtree   : Decision Trees
*    forest  : Random Forests
*    extra   : Extra Trees
*    kmp     : KMeans->LogisticRegression Pipelines
*    kernp   : Nystroem Approx->Logistic Regression Pipelines

Some model choices are __very slow__.  The default is to use decision trees, which are reasonably fast.

The simplest command line is:

    unix> ./ensemble_train.py some_dbfile.db some_data.svm

__*(NOTE: Expects 'some_dbfile.db' not to exist, and will quit if it does [so you don't accidentally blow away your model].)*__
    
Full usage is:

```
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
```



#### __ensemble_predict.py__

Get predictions from trained EnsembleSelectionClassifier given
svm format data file.

Can output predicted classes or probabilities from the full
ensemble or just the best model.

Expects to find a trained ensemble in the sqlite db specified.

```
usage: ensemble_predict.py [-h] [-s {best,ens}] [-p] db_file data_file

Get EnsembleSelectionClassifier predictions

positional arguments:
  db_file        sqlite db file containing model
  data_file      testing data in svm format

optional arguments:
  -h, --help     show this help message and exit
  -s {best,ens}  choose source of prediction ["best", "ens"]
  -p             predict probabilities
```

Requirements
------------

Written using Python 2.7.3, numpy 1.6.1, scipy 0.10.1, scikit-learn 0.14.1 and sqlite 3.7.14


References
----------
[1] Caruana, et al, "Ensemble Selection from Libraries of Rich Models", Proceedings of the 21st International Conference on Machine Learning (ICML `04).
    
[2] Caruana, et al, "Getting the Most Out of Ensemble Selection", Proceedings of the 6th International Conference on Data Mining (ICDM `06).
    

