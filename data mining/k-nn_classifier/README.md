1_k-nn_classifier_jt_01-25-2022
==============================

- __Description:__ Implementation of a straightforward classification algorithm known as the k-nearest neighbor (k-NN)
  classifier.
- __Iris Data Set:__ contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class
  is linearly separable from the other 2; the latter are NOT linearly separable from each other.
- __Predicted attribute:__ class of iris plant.flowers dataset.

How to run the code
------------

- The train.csv and test.csv must be in /data/raw/ folder.
- Run each .py file on this order:

1. Run 'make_dataset.py' in /src/data/make_dataset.py to:
    - Return Data transformed from the raw data, to proceed with the analysis later on.
    - Store transformed data it into the /interim folder.
2. Run 'build_features.py' in /src/features/build_features.py to:
    - Return the processed data sets, by:
        - splitting them into:
            - train, validation sets
            - x (predictors) and y (target variable, encoded)

        - normalizing and formatting:
            - train, validation and test sets,
            - to properly train the KNN Classifier later on.
3. Run 'train_model.py' in /src/models/train_model.py to:
    - Train model for each k-neighbors with both:
        - KNN model from sklearn
        - KNN model from scratch
4. Run 'predict_model.py' in /src/models/predict_model.py to:
    - Measure the accuracy of the model with the validation data. For the 2 models:
        - KNN from scratch.
        - KNN from sklearn.
    - Make the predictions with the test data, and build a .csv file with the requested and decoded format of the target
      variable, taking into account each k-neighbors.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── output         <- The final .csv file requested for the client.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionary.
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
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │  
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

Results
------------

- The classification done follows the following requested requirements:
    - The classification is based on an unweighted vote of its k-nearest examples in the training set.
    - To determine nearest neighbors, all distances using regular Euclidean distance have been measured.
    - If two or more classes receive the same (winning) number of votes, break the tie by choosing the class with the
      lowest total distance from the test point to its voting examples.
        - This has been done for the knn implementation from scratch, and it is the default break tie of sklearn.
- In order to measure the performance of the model:
    - The x predictors have been Normalized, to avoid that when one feature values are larger than other, that feature
      dominates the distance needed in the KNN algorithm.
    - The training data has been Split into:
        - Training set (x predictors and target y), 80% of the Training data.
        - Validation set (x predictors and target y), 20% of the Training data.
    - The performance of the model seems to be great when training the model with the Training data. With an accuracy of
      a 100%, for different k-neighbors settings.
- Predictions have been done with the out of sample Test Data.
- An Output.csv is generated showing the results of the predictions for each k-neighbors.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
