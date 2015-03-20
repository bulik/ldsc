'''
A decorator that parses docstrings and checks the shapes of array arguments.

'''
import re
import itertools as it
import inspect

array_re = re.compile(' : np.ndarray with shape \(.*,.*\)')


def count_leading_spaces(line):
    '''Count leading spaces.'''
    return len(line) - len(line.lstrip(' '))


def base_indent(lines):
    '''Returns the lines at base indentation level.'''
    indent_levels = [count_leading_spaces(line) for line in lines]
    base_indent_level = min(indent_levels)
    base_indent_lines = [line for i, line in enumerate(lines)
                         if indent_levels[i] == base_indent_level]
    return base_indent_lines


def filter_colon(lines):
    '''Returns lines that have a ' : ' pattern.'''
    filtered_lines = [line for line in lines if ' : ' in line]
    return filtered_lines


def param_to_returns(doc):
    '''Returns the lines in doc between 'Parameters' and 'Returns'.'''
    lines = doc.split('\n')
    try:
        start = lines.index('Parameters')
    except ValueError:
        raise ValueError(
            'Could not find a line labeled Parameters in doctring,')
    try:
        end = lines.index('Returns')
    except ValueError:
        raise ValueError('Could not find a line labeled Returns in docstring,')
    if start > end:
        raise ValueError('Returns came before Parameters in docstring.')
    return lines[start:end]


def parse_shape(line):
    '''Extract shape from an arg spec. Return None if arg is not an array.'''
    if re.search(array_re, line) is not None:
        parens = re.search('\(.*?,.*?\)', line).group(0)
        parens = parens.replace(' ', '')
        parens = parens[1:-1]
        shape = parens.split(',')
        for i, v in enumerate(shape):
            try:
                shape[i] = int(v)
            except ValueError:
                continue
        if shape[-1] == '' and len(shape) == 2:  # catch shapes like (1, )
            shape.pop()
        elif shape[-1] == '' and len(shape) != 2:  # (1, 1, ) is invalid
            raise ValueError(
                'Shapes ending with ,) are only valid for 1D arrays.')
        return shape
    else:
        return None


def docshapes(f):
    '''
    Decorator that parses docstring and check array shapes.

    Usage
    -----
    The docshapes decorator will parse the docstring of the decorated function.
    The docstring must therefore be formatted according to certain rules. Lines
    declaring the shapes of arrays must be at the base indent level for the
    docstring and contain the substring ' : np.ndarray with shape [shape]',
    where [shape] is a shape expressed as (a, b), etc. Lines without array
    shape declarations will be ignored. Lines that do not contain argument
    declarations must be indented at a higher level than the base indent level
    for the docstring.

    All input parameter declarations must lie between lines containing
    'Parameters' and 'Returns'. Optional arguments (e.g., x=None by default
    but an (n, p) array otherwise) are handled correctly.

    '''

    doc = inspect.cleandoc(inspect.getdoc(f))
    lines = param_to_returns(doc)
    def_lines = filter_colon(base_indent(lines))
    # None if arg is not an array
    shapes = [parse_shape(line) for line in def_lines]

    def wrapped_f(*args, **kwargs):
        dims = {}
        f_sig = inspect.getargspec(f)
        z = zip(args, shapes, it.count(), f_sig.args)
        print z
        try:
            for a, t, n, a_name in z:
                exp_shape = shapes[n]
                if exp_shape is None:
                    continue
                try:
                    cur_shape = a.shape
                except AttributeError:
                    raise AttributeError('%s is not an array.' % a_name)

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
