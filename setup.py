from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='ldsc',
      version='2.0.1',
      description='LD Score Regression (LDSC)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='http://github.com/bulik/ldsc',
      author='Brendan Bulik-Sullivan and Hilary Finucane',
      author_email='',
      license='GPLv3',
      packages=['ldscore'],
      scripts=['ldsc.py', 'munge_sumstats.py', 'make_annot.py'],
      install_requires = [
            'bitarray>=2.6.0',
            'nose>=1.3.7', # this is only a test dep
            'pybedtools>=0.9.0',
            'scipy>=1.9.2',
            'numpy>=1.23.3',
            'pandas>=1.5.0'
      ]
)
