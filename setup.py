"""Setup file

Tutorial:
http://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html
"""

from distutils.core import setup
import pymlearn

setup(
	name="pymlearn",
	version=pymlearn.__version__,
	author_email="xin.heng@gmail.com",
	url="",
	packages=["pymlearn", ],
	license=""
)
