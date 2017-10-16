## pymlearn: python ML functions

*
* I mostly adhere to scikit-learn's API design

## Installation

### The clone repo way
```
$ make clean-build
$ python setup.py sdist
$ pip install [--user] [-t python-lib-location] dist/pymlearn-<version>.tar.gz
```

* "-t" allows you to install at a specified location
* "--user" is a mac os feature, allowing you to install at user level
* The conventional way is to install it in a virtual environment

### The github way

`$ pip install -e git+git@github.com:xinh3ng/pymlearn.git#egg=< branchname >`

## Examples
```
```

## TODO (ideas in mind)
`
R functions that are superior to Python: auto.arima
`