import re
import itertools as it
import inspect

'''
Parameters
----------
Each argument should be a tuple (variable, string), where variable
is a variable passed to the parent argument and string defines the
shape of a numpy array. Shapes can be defined with the following syntax:
(a, ) 1D array with length a
(a, b) 2D array with dimensions a, b
(1, b) 2D array with shape (1, b)
* is interpreted as a free dimension
... represents 0+ dimensions
letters are interpreted as variables, and are assigned values according
to the shapes of the arguments, from left to right.
### TODO add support for ellipses
(a, b, ...) n-D array with first two dimensions equal to a, b
    (the ellipsis must come at either the beginning or the end)
### TODO add support for optional arguments e.g., shape (a, b) or None
'''


def subset_docstring(docs):
    '''
    Returns the lines in docs between 'Parameters' and 'Returns' that match
    the regex ' : np.ndarray with shape (.*,.*)'
    '''
    lines = docs.split('\n')
    try:
        start = lines.index('Parameters')
    except ValueError:
        raise ValueError(
            'Could not find a line labeled Parameters in docstring,')
    try:
        end = lines.index('Returns')
    except ValueError:
        raise ValueError('Could not find a line labeled Returns in docstring,')

    if start > end:
        raise ValueError('Returns came before Parameters in docstring.')

    arrays = re.compile(' : np.ndarray with shape \(.*,.*\)')
    lines = filter(
        lambda z: re.search(arrays, z) is not None, lines[start:end])
    return lines


def get_shapes(lines):
    '''
    lines is a tuple of lines all of which match the regex
    ' : np.ndarray with shape (.*?, .*?)'
    This function returns the shapes.
    Note that this will give confusing errors if the docstring specifies variables
    out of order.
    '''
    shapes = []
    for line in lines:
        parens = re.search('\(.*?,.*?\)', line).group(0)
        parens = parens.replace(' ', '')
        parens = parens[1:-1]
        vals = parens.split(',')
        for i, v in enumerate(vals):
            try:
                vals[i] = int(v)
            except ValueError:
                continue

        shapes.append(vals)

    for s in shapes:
        if s[-1] == '' and len(s) == 2:  # catch shapes like (1, )
            s.pop()
        # complain about shapes like (1, 1, )
        elif s[-1] == '' and len(s) != 2:
            raise ValueError(
                'Shapes ending with ,) are only valid for 1D arrays.')

    return shapes


def docshapes(f):
    doc = inspect.cleandoc(inspect.getdoc(f))
    lines = subset_docstring(doc)
    shapes = get_shapes(lines)

    def wrapped_f(*args, **kwargs):
        dims = {}
        f_sig = inspect.getargspec(f)
        z = zip(args, shapes, it.count(), f_sig.args)
        try:
            for a, t, n, a_name in z:
                '''
                Note that the above iterator works even if f has *args and **kwargs
                because of the restriction that *args and **kwargs come after the
                primary arguments. If f has *args, then args will be longer than
                shapes and f_sig.args, but the output of the zip function is only
                as long as the shortest of its argumetns, so this is OK.
                Right now it only checks the required arguments, not the *args and **kwargs.
                TODO: check the *args and **kwargs
                '''
                try:
                    cur_shape = a.shape
                except AttributeError:
                    raise AttributeError('%s is not an array.' % a_name)

                exp_shape = shapes[n]
                cur_dim = len(cur_shape)
                exp_dim = len(exp_shape)
                if cur_dim != exp_dim:
                    raise TypeError(
                        'Argument %s has %d dimensions, should have %d.' % (a_name, cur_dim, exp_dim))
                for i, d in enumerate(exp_shape):
                    if isinstance(d, (int, long)):
                        if cur_shape[i] != d:
                            raise TypeError(
                                'Argument %s has shape %s; dimension %s should be %s.' % (a_name, cur_shape, i, d))
                    elif d in dims:
                        if cur_shape[i] != dims[d]:
                            raise TypeError(
                                'Argument %s has shape %s; dimension %s should be %s.' % (a_name, cur_shape, i, dims[d]))
                    else:
                        dims[d] = cur_shape[i]

        except TypeError as e:
            m = 'Shapes incompatible with docstring.\nShapes of arguments to %s()\n' % f.__name__
            s = dict()
            for a, t, n, a_name in z:
                try:
                    s = str(a.shape)
                except AttributeError:
                    s = 'not an array'
                m += '    %s: %s\n' % (a_name, s)

            m += str(e.args[0])
            raise TypeError(m)

        return f(*args, **kwargs)
    return wrapped_f
