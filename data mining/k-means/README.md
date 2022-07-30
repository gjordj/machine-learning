k_means
==============================

Implementation of the K-Means algorithm from scratch and using sklearn, in order to compress an image and recolor it 
based on the k-clusters.

How to run the code
------------
1. Run make_dataset.py to load the image to compress.
2. Run train_model_1.py: this will compress the image with initialized centroids.
   - However, the only given centroid 
      that worked well, was the first one, so a random initialization has been done, although the code with the given 
      centroids can be tried, removing the comment format.
3. (Optional), Run train_model_2_sklearn: same but with the sklearn library.

Results
------------
- In the "reports/figures" folder:
  - compressed_image_2.png
  - compressed_image_4.png
  - compressed_image_7.png
  - compressed_image_10.png


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
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
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── train_model_2_sklearn.py
    │   │   └── train_model_1.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
