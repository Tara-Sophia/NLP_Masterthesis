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
conda env update --name nlp_masterthesis --file environment.yml --prune
```

---
To *create a new version of the environment.yml file*

```bash
conda env export | grep -v "^prefix: " > environment.yml
```

Pip installation:

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

## Documentation

To update the documenation run the following commands:

```bash
cd docs
```

```bash
sphinx-apidoc -o ./source ../src --separate --force
```

```bash
make clean
```

```bash
make html
```

## Contributing

This project is build by three students from [Nova SBE](https://www.novasbe.unl.pt/en/) in collaboration with  [Assistant Professor Qiwei Han](https://www.novasbe.unl.pt/en/faculty-research/faculty/faculty-detail/id/137/qiwei-han)

- Hannah Petry (48458@novasbe.pt)
- Tara-Sophia Tumbraegel (48333@novasbe.pt)
- Florentin von Haugwitz (48174@novasbe.pt)

## License

[MIT](https://choosealicense.com/licenses/mit/)
