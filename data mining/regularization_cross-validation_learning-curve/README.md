regularization_cross-validation_learning-curve
==============================

Implementation of L2 regularization (Ridge) linear regression algorithm, cross-validation, learning curve.
Calculation of MSE and learning curve based on lambda, training size.

A report is in the folder /reports.

Running Code
------------
- Go to /src/data/ and run make_dataset.py to generate additional files.
- Go to /src/models/ and run train_model.py to make the analysis and generate results.
  - Ridge Regression can be trained by using sklearn or code from scratch by changing ridge_scratch variable to 
    True/False, in the main() of the train_model.py.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── output      <- .csv file with the differences between MSE and lambdas.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
