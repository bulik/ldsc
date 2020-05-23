from setuptools import setup

setup(name='ldsc',
      version='1.0',
      description='LD Score Regression (LDSC)',
      url='http://github.com/bulik/ldsc',
      author='Brendan Bulik-Sullivan and Hilary Finucane',
      author_email='',
      license='GPLv3',
      packages=['ldscore'],
      scripts=['ldsc.py', 'munge_sumstats.py'],
      install_requires = [
            'bitarray>=0.8,<0.9',
            'nose>=1.3,<1.4',
            'pybedtools>=0.7,<0.8',
            'scipy>=0.18,<0.19',
            'numpy>=1.16,<1.17',
            'pandas>=0.20,<0.21'
      ]
)
