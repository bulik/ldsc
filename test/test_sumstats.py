from __future__ import division
import ldscore.sumstats as s
import unittest
import numpy as np
import pandas as pd
import nose
from nose_parameterized import parameterized as param
from pandas.util.testing import assert_series_equal
from pandas.util.testing import assert_frame_equal


class Mock(object):
	'''
	Dumb object for mocking args and log
	'''
	def __init__(self):
		pass
	
	def log(self, x):
		pass
		
log = Mock()
args = Mock()