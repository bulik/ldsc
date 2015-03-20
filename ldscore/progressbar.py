'''
(c) 2015-Present Brendan Bulik-Sullivan and Hilary Finucane
A simple progressbar.

'''

from __future__ import division
import sys


def update_progress(progress, bar_length=40):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        status = "error: progress var must be float\r\n"
    if progress < 0:
        status = "Halt...\r\n"
    if progress >= 1:
        text = "\r{:}100%\n".format("."*bar_length)
        sys.stdout.write(text)
        sys.stdout.flush()
    else:
        block = int(round(bar_length*progress))
        text = "\r{:}{:.0%} {:}".format("."*block + " "*(bar_length-block), progress, status)
        sys.stdout.write(text)
        sys.stdout.flush()
