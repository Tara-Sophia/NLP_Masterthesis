# NLP_Masterthesis

This project contributes to the public health sector by building a natural language processing system to interact with a patient via a chatbot to give the most accurate diagnosis possible.

## Installation

Use the package manager conda to install the enviroment

```bash
conda env create -f environment.yml
```

---
To create a new version of the enviroment

- Mac/Linux

```bash
conda env export --from-history | grep -v "^prefix: " > environment.yml
```

- Windows

```bash
conda env export --from-history | findstr -v "^prefix: " > environment.yml
```

## Contributing

This project is build by three students from [Nova SBE](https://www.novasbe.unl.pt/en/) in collaboration with  [Assistant Professor Qiwei Han](https://www.novasbe.unl.pt/en/faculty-research/faculty/faculty-detail/id/137/qiwei-han)

- Hannah Petry (48458@novasbe.pt)
- Tara-Sophia Tumbraegel (48333@novasbe.pt)
- Florentin von Haugwitz (48174@novasbe.pt)

## License

[MIT](https://choosealicense.com/licenses/mit/)
