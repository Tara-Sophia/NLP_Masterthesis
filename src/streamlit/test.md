├── data
│   ├── interim                             # Folder for all intermediate data that has been transformed
│   │
│   ├── processed                           # Folder for all the cleaned data sets for modeling
│   │
│   └── raw                                 # Folder for the original, immutable data dump
│
├── docs                                    # Folder for all documentation files for the project
│
├── models
│   ├── clf                                 # Folder for all classifier models
│   │
│   ├── nlp                                 # Folder for all Natural Language Processing models
│   │
│   └── stt                                 # Folder for all Speech to Text models
│
├── notebooks                               # Folder for all Jupyter notebooks to explore the data and models
│
├── src                                     # Main code for the project
│   ├── clf                                 # Folder for all classifier scripts
│   │   ├── constants.py                    # Python script to store all constants for the classification models
│   │   ├── create_data.py                  # Python script to create the data for the classification models
│   │   ├── *_training.py                   # Python script to train the classification models dependent on the model used
│   │   ├── *_evaluate.py                   # Python script to evaluate the classification models dependent on the model used
│   │   ├── predict.py                      # Python script to predict an example with a trained classification model
│   │   └── utils.py                        # Python script to store all utility functions for the classification models
│   │
│   ├── icd_mimic                           # Folder for all scripts to create the joined ICD-MIMIC dataset
│   │   ├── constants.py                    # Python script to store all constants for the joined ICD-MIMIC dataset
│   │   └── prepare_mimic.py                # Python script to create the joined ICD-MIMIC dataset
│   │
│   ├── nlp                                 # Folder for all Natural Language Processing scripts
│   │   ├── constants.py                    # Python script to store all constants for the NLP models
│   │   ├── create_data.py                  # Python script to create the data for the NLP models
│   │   ├── *_training.py                   # Python script to train the NLP models dependent on the model used
│   │   ├── *_evaluation.py                 # Python script to evaluate the NLP models dependent on the model used
│   │   ├── predict.py                      # Python script to predict an example with a trained NLP model
│   │   └── utils.py                        # Python script to store all utility functions for the NLP models
│   │
│   ├── streamlit                           # Folder for all Streamlit scripts
│   │   ├── app_clf.py                      # Python script with the classification model for the streamlit app
│   │   ├── app_nlp.py                      # Python script with the NLP model for the streamlit app
│   │   ├── app_stt.py                      # Python script with the STT model for the streamlit app
│   │   └── app.py                          # Python script to execute the streamlit app
│   │
│   └── stt                                 # Folder for all Speech to Text scripts
│       ├── constants.py                    # Python script to store all constants for the STT models
│       ├── create_data.py                  # Python script to create the data for the STT models
│       ├── *_training.py                   # Python script to train the STT models dependent on the model used
│       ├── *_evaluation.py                 # Python script to evaluate the STT models dependent on the model used
│       ├── predict.py                      # Python script to predict an example with a trained STT model
│       └── utils.py                        # Python script to store all utility functions for the STT models
│
└── tests                                   # Pytests to validate the project