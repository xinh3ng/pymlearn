"""Setup file

Tutorial:
http://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html
"""

# from distutils.core import setup  # This is for sdist
from setuptools import setup  # This is for bdist_wheel
import pymlearn

setup(
	name="pymlearn",
	version=pymlearn.__version__,
	author_email="district24x7@gmail.com",
	url="",
	packages=["pymlearn", ],
	license=""
)
