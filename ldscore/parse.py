'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This module contains functions for parsing various ldsc-defined file formats.

'''

from __future__ import division
import numpy as np
import pandas as pd
import os
import re
arr_re = re.compile('\{\d+:\d+\}')
arr_left_re = re.compile('\{\d+:')
arr_right_re = re.compile(':\d+\}')


def exp_array(fh):
    '''Process array notation {1:22} in filenames.'''
    arr = arr_re.findall(fh)
    if len(arr) > 1:
        raise ValueError('Can only have one array {a:b} per filename.')
    elif len(arr) == 0:
        return [fh]
    else:
        lo = int(arr_left_re.search(fh).group(0)[1:-1])
        hi = int(arr_right_re.search(fh).group(0)[1:-1])
        return [arr_re.sub(str(i), fh) for i in xrange(lo, hi)]


def series_eq(x, y):
    '''Compare series, return False if lengths not equal.'''
    return len(x) == len(y) and (x == y).all()


def read_csv(fh, **kwargs):
    return pd.read_csv(fh, delim_whitespace=True, na_values='.',
                       comment='#', **kwargs)


def sub_chr(s, chr):
    '''Substitute chr for @, else append chr to the end of str.'''
    if '@' not in s:
        s += '@'
    return s.replace('@', str(chr))


def which_compression(fh):
    '''Given a file prefix, figure out what sort of compression to use.'''
    if os.access(fh + '.bz2', 4):
        suffix = '.bz2'
        compression = 'bz2'
    elif os.access(fh + '.gz', 4):
        suffix = '.gz'
        compression = 'gzip'
    elif os.access(fh, 4):
        suffix = ''
        compression = None
    else:
        raise IOError('Could not open {F}[./gz/bz2]'.format(F=fh))
    return suffix, compression


def get_compression(fh):
    '''Which sort of compression should we use with read_csv?'''
    if fh.endswith('gz'):
        compression = 'gzip'
    elif fh.endswith('bz2'):
        compression = 'bz2'
    else:
        compression = None
    return compression


def read_cts(fh, match_snps):
    '''Reads files for --cts-bin.'''
    compression = get_compression(fh)
    cts = read_csv(fh, compression=compression, header=None, names=['SNP', 'ANNOT'])
    if not series_eq(cts.SNP, match_snps):
        raise ValueError('--cts-bin and the .bim file must have identical SNP columns.')

    return cts.ANNOT.values


def sumstats(fh, alleles=False, dropna=True):
    '''Parses .sumstats files. See docs/file_formats_sumstats.txt.'''
    dtype_dict = {'SNP': str,   'Z': float, 'N': float, 'A1': str, 'A2': str}
    compression = get_compression(fh)
    usecols = ['SNP', 'Z', 'N']
    if alleles:
        usecols += ['A1', 'A2']
    try:
        x = read_csv(fh, usecols=usecols, dtype=dtype_dict, compression=compression)
    except (AttributeError, ValueError) as e:
        raise ValueError('Improperly formatted sumstats file: ' + str(e.args))
    if dropna:
        x = x.dropna(how='any')
    return x


def read_fromlist(flist, parsefunc, noun, *args, **kwargs):
    '''Sideways concatenation. *args and **kwargs are passed to parsefunc.'''
    df_array = []
    for i, fh in enumerate(flist):
        y = parsefunc(fh, *args, **kwargs)
        if i > 0:
            if not series_eq(y.SNP, df_array[0].SNP):
                raise ValueError('%s files for concatenation must have identical SNP columns.' % noun)
            else:  # keep SNP column from only the first file
                y = y.drop(['SNP'], axis=1)
        new_col_dict = {c: c + '_' + str(i) for c in y.columns if c != 'SNP'}
        y.rename(columns=new_col_dict, inplace=True)
        df_array.append(y)
    return pd.concat(df_array, axis=1)


# ldscore

def ldscore_parser(fh, compression):
    '''Parse LD Score files'''
    x = read_csv(fh, header=0, compression=compression)
    if 'MAF' in x.columns and 'CM' in x.columns:  # for backwards compatibility w/ v<1.0.0
        x = x.drop(['MAF', 'CM'], axis=1)
    return x


def ldscore(fh):
    '''Parse .l2.ldscore files, split across num chromosomes. See docs/file_formats_ld.txt.'''
    suffix = '.l2.ldscore'
    fhs = exp_array(fh)
    chr_ld = []
    for fh in fhs:
        full_fh = fh + suffix
        s, compression = which_compression(full_fh)
        chr_ld.append(ldscore_parser(full_fh + s, compression))
    x = pd.concat(chr_ld)  # automatically sorted by chromosome
    x = x.sort(['CHR', 'BP'])  # SEs will be wrong unless sorted
    x = x.drop(['CHR', 'BP'], axis=1).drop_duplicates(subset='SNP')
    return x


def ldscore_fromlist(flist):
    return read_fromlist(flist, ldscore, 'LD Score')


# M / M_5_50

def M_parser(fh):
    '''Parse a .l2.M or .l2.M_5_50 file.'''
    return [float(z) for z in open(fh, 'r').readline().split()]


def M(fh, N=2, common=False):
    '''Parses .M* files.'''
    suffix = '.l' + str(N) + '.M'
    if common:
        suffix += '_5_50'
    fhs = exp_array(fh+suffix)
    x = np.sum((M_parser(fh) for fh in fhs), axis=0)
    return np.array(x).reshape((1, len(x)))


def M_fromlist(flist, N=2, common=False):
    '''Read a list of .M* files and concatenate sideways.'''
    return np.hstack([M(fh, N, common) for fh in flist])


# annot / frqfile

def annot_parser(fh, compression, frqfile=None, compression_frq=None):
    '''Parse annot files'''
    df_annot = read_csv(fh, header=0, compression=compression).drop(['CHR', 'BP', 'CM'], axis=1)
    df_annot.iloc[:, 1:] = df_annot.iloc[:, 1:].astype(float)
    if frqfile is not None:
        df_frq = frq_parser(frqfile, compression_frq)
        if not series_eq(df_frq.SNP, df_annot.SNP):
            raise ValueError('.frqfile and .annot must have the same SNPs in same order.')
        df_annot = df_annot[(.95 > df_frq.FRQ) & (df_frq.FRQ > 0.05)]
    return df_annot


def frq_parser(fh, compression):
    '''Parse frequency files.'''
    df = read_csv(fh, header=0, compression=compression)
    if 'MAF' in df.columns:
        df.rename(columns={'MAF': 'FRQ'}, inplace=True)
    return df[['SNP', 'FRQ']]


def annot(fh_list, num=None, frqfile=None):
    '''Parses .annot files and returns an overlap matrix. '''
    annot_suffix = ['.annot' for fh in fh_list]
    annot_compression = []
    if num is not None:  # 22 files, one for each chromosome
        for i, fh in enumerate(fh_list):
            first_fh = sub_chr(fh, 1) + annot_suffix[i]
            annot_s, annot_comp_single = which_compression(first_fh)
            annot_suffix[i] += annot_s
            annot_compression.append(annot_comp_single)
        if frqfile is not None:
            frq_suffix = '.frq'
            first_frqfile = sub_chr(frqfile, 1) + frq_suffix
            frq_s, frq_compression = which_compression(first_frqfile)
            frq_suffix += frq_s
        y = []
        M_tot = 0
        for chr in xrange(1, num + 1):
            if frqfile is not None:
                df_annot_chr_list = [annot_parser(sub_chr(fh, chr) + annot_suffix[i], annot_compression[i],
                                                  sub_chr(frqfile, chr) + frq_suffix, frq_compression)
                                     for i, fh in enumerate(fh_list)]
            else:
                df_annot_chr_list = [annot_parser(sub_chr(fh, chr) + annot_suffix[i], annot_compression[i])
                                     for i, fh in enumerate(fh_list)]
            annot_matrix_chr_list = [np.matrix(df_annot_chr.ix[:, 1:]) for df_annot_chr in df_annot_chr_list]
            annot_matrix_chr = np.hstack(annot_matrix_chr_list)
            y.append(np.dot(annot_matrix_chr.T, annot_matrix_chr))
            M_tot += len(df_annot_chr_list[0])
        x = sum(y)
    else:  # just one file
        for i, fh in enumerate(fh_list):
            annot_s, annot_comp_single = which_compression(fh + annot_suffix[i])
            annot_suffix[i] += annot_s
            annot_compression.append(annot_comp_single)
        if frqfile is not None:
            frq_suffix = '.frq'
            frq_s, frq_compression = which_compression(frqfile + frq_suffix)
            frq_suffix += frq_s
            df_annot_list = [annot_parser(fh + annot_suffix[i], annot_compression[i],
                                          frqfile + frq_suffix, frq_compression) for i, fh in enumerate(fh_list)]
        else:
            df_annot_list = [annot_parser(fh + annot_suffix[i], annot_compression[i])
                             for i, fh in enumerate(fh_list)]
        annot_matrix_list = [np.matrix(y.ix[:, 1:]) for y in df_annot_list]
        annot_matrix = np.hstack(annot_matrix_list)
        x = np.dot(annot_matrix.T, annot_matrix)
        M_tot = len(df_annot_list[0])

    return x, M_tot


def annot_fromlist(flist, frqfile=None):
    return read_fromlist(flist, annot, 'Annot', frqfile=frqfile)
