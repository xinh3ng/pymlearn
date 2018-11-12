install:
	pip install pipenv
	pipenv run pip install pip==18.0
	pipenv install --skip-lock

lint:
	pipenv run black pymlearn tests -l 120 --py36

lint-check:
	pipenv run black pymlearn tests -l 120 --py36 --check

test:
	pipenv run pytest -s -v --cov=pymlearn tests --cov-fail-under=5 --disable-pytest-warnings

