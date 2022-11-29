# Github components

## Sphinx Documentation

The reason for having a documentation is to provide a reference for the user to understand the code and the project. The documentation is written in Sphinx, which is a documentation generator that uses reStructuredText as its markup language. The documentation is hosted on Github pages, which is a documentation hosting service. The documentation is automatically updated and deployed on every push to the master branch of the repository.

Refrences: [Sphinx](https://www.sphinx-doc.org/en/master/), [Read the Docs](https://readthedocs.org/)

## Tests

Tests are used to ensure that the code is working as intended. This is done by running the code on a set of inputs and comparing the output to the expected output. The tests are written in pytest, which is a testing framework for Python. The tests are run on every commit to the repository, whereas the commmit fails if the tests fail. 

The tests are also run on every pull request to the repository, and the pull request fails if the tests fail.


### Conda environment

A conda environment is used to ensure that the code is run in the same environment as the one used to develop the code. This is done by creating a conda environment file, which is a file that contains the dependencies of the project. The dependencies are installed in the conda environment by running the command `conda env create -f environment.yml`. The conda environment is activated by running the command `conda activate <environment_name>`. The conda environment is deactivated by running the command `conda deactivate`. The advantage of using a conda environment is that it is easy to install the dependencies, and it is easy to activate and deactivate the environment.

References: [Conda](https://docs.conda.io/en/latest/), [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

A virtual environment creates 