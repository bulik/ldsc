try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'LD SCore (LDSC)',
    'author': 'Brendan Bulik-Sullivan and Hilary Finucane',
    'url': 'https://bitbucket.org/bulik/ld-score',
    'download_url': '',
    'author_email': 'bulik@broadinstitute.org',
    'version': '0.0001',
    'install_requires': ['nose', 'scipy.spstats', 'numpy', 'progressbar', 'bitarray', 'nose_parameterized'],
    'packages': ['ldsc'],
    'scripts': ['ldsc.py'],
    'name': 'LDSC'
}

setup(**config)


