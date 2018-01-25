## pymlearn

Python Machine Learning Functions

* I mostly adhere to scikit-learn's API design (and I do use a lot of sklearn). 
* I make changes at places where either I want more control or I deviate from sklearn's API design.

## Installation

### The clone repo way

`$ python setup.py sdist`

`$ pip install [--user] [-t python-lib-location] dist/pymlearn-< version >.tar.gz`  
* "-t" allows you to install at a specified location
* "--user" is a mac os feature, allowing you to install at user level
* The conventional way is to install it in a virtual environment

### The github way

`$ pip install -e git+git@github.com:xinh3ng/pymlearn.git#egg=< branchname >`

## Examples
```
```

## TODO (ideas in mind)

* Make my API compatible with both pandas and pyspark
* Enable parallel computing at several time-consuming steps
* Bring in some R functionalities that are superior to Python counterparts, e.g. auto.arima
* ...
