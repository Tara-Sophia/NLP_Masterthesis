# NLP_Masterthesis

This project contributes to the public health sector by building a natural language processing system to interact with a patient via a chatbot to give the most accurate diagnosis possible.

## Installation

Use the package manager conda to *install the environment*

```bash
conda env create -f environment.yml
```

---
To *update the existing environment*

```bash
conda env update --file environment.yml
```

---
To *create a new version of the environment.yml file*

```bash
conda env export --from-history | grep -v "^prefix: " > environment.yml
```

## Directory  Structure

```text
├── .github                                 # Templates and workflows for PRs
│
├── data
│   ├── external                            # Data from third party sources
│   │
│   ├── interim                             # Intermediate data that has been transformed
│   │
│   ├── processed                           # The final, canonical data sets for modeling
│   │
│   └── raw                                 # The original, immutable data dump
│
├── docs                                    # A default Sphinx project
│
├── models                                  # Trained and serialized models, model predictions, or model summaries
│
├── notebooks                               # Jupyter notebooks
│
├── references                              # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                                 # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                             # Generated graphics and figures to be used in reporting
│
├── src                                     # Code for the project
│   ├── __init__.py                         # Makes src a Python module
│   │
│   ├── data                                # Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── models                              # Scripts to train models and then use trained models to make predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   ├── transformation                      # Scripts to transform the data and create features
│   │   └── make_transformations.py
│   │
│   └── visualization                       # Scripts to make visualizations
│       └── visualize.py
│
├── tests                                   # Tests for the project
│   ├── __init__.py                         # Makes tests a Python module
│   │
│   ├── data                                # Scripts to test data folder
│   │   └── test_make_dataset.py
│   │
│   ├── models                              # Scripts to test models folder
│   │   ├── test_predict_model.py
│   │   └── test_train_model.py
│   │
│   ├── transformation                      # Scripts to test transformation folder
│   │   └── test_make_transformations.py
│   │
│   └── visualization                       # Scripts to test visualization folder
│       └── test_visualize.py
│
├── .gitignore                              # File listing names of files Git should ignore
├── environment.yml                         # File to reproduce environment. For installation look above         
├── LICENSE                                 # MIT License         
└── README.md                               # Top-level README containing usage and description of project
```

## Contributing

This project is build by three students from [Nova SBE](https://www.novasbe.unl.pt/en/) in collaboration with  [Assistant Professor Qiwei Han](https://www.novasbe.unl.pt/en/faculty-research/faculty/faculty-detail/id/137/qiwei-han)

- Hannah Petry (48458@novasbe.pt)
- Tara-Sophia Tumbraegel (48333@novasbe.pt)
- Florentin von Haugwitz (48174@novasbe.pt)

## License

[MIT](https://choosealicense.com/licenses/mit/)
