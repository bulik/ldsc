'''
(c) 2015-Present Brendan Bulik-Sullivan and Hilary Finucane
A simple progressbar.

'''

from __future__ import division
import sys
import time


def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=int(d))
    if h > 0:
        f += '{H}h:'.format(H=int(h))
    if m > 0:
        f += '{M}m:'.format(M=int(m))

    f += '{S}s'.format(S=round(s, 1))
    return f


class Progress(object):
    '''
    A progressbar.

    Parameters
    ----------
    end : float
        Max count.
    bar_len : int, optional
        Progress bar length.
    desc : str (optional)
        Description.

    Attributes
    ----------
    end : float
        Max count.
    progress : float
        Current count.
    start_time : float
        Start time.
    current_time : float
        Time of last call to _eta().
    last_len : int
        Length of last string printed to stdout (for clearning previous line).
    desc : str
        Description.

    Methods
    -------
    update_progress(progress)
        Update progressbar.
    _eta :
        Calculate ETA.

    '''

    def __init__(self, end, bar_len=40, desc=''):
        self.progress = 0
        self.end = end
        self.start_time = None
        self.current_time = None
        self.bar_len = bar_len
        self.last_len = 0
        self.desc = desc

    def _eta(self):
        '''Estimate ETA via linear extrapolation.'''
        self.current_time = time.time()
        self.time_elapsed = self.current_time - self.start_time
        time_per_unit = self.time_elapsed / self.progress
        units_remaining = self.end - self.progress
        eta = time_per_unit * units_remaining
        return eta

    def update_progress(self, progress, bar_length=40):
        '''Update progressbar'''
        if self.start_time is None:
            self.start_time = time.time()
            self.current_time = self.start_time
        self.progress = float(progress)
        sys.stdout.write('\r'' '*(self.last_len+2))
        sys.stdout.flush()
        if self.progress < 0:
            raise ValueError('progress must be > 0.')
        block = int(bar_length*self.progress/self.end)
        text = '\r'+self.desc+' %s%%|' % str(int(self.progress/self.end*100))
        text += '%s|' % ('#'*block+' '*(bar_length-block))
        if self.progress > 0:
            text += ' ETA %s ' % sec_to_str(self._eta())
            text += 'Elapsed %s ' % sec_to_str(self.time_elapsed)
        self.last_len = len(text)
        sys.stdout.write(text)
        sys.stdout.flush()
