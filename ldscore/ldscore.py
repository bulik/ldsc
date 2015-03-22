from __future__ import division
import numpy as np
import bitarray as ba
from docshapes import docshapes
from progressbar import Progress
import parse as ps
import plink


def to_csv(fh, **kwargs):
    return pd.to_csv(fh, index=False, header=True, sep='\t', **kwargs)


pd.DataFrame.to_csv2 = to_csv


def block_lefts(coords, max_dist):
    '''
    Converts coordinates + max block length to the a list of coordinates of the leftmost
    SNPs to be included in blocks.

    Parameters
    ----------
    coords : array
        Array of coordinates. Must be sorted.
    max_dist : float
        Maximum distance between SNPs included in the same window.

    Returns
    -------
    block_left : 1D np.ndarray with same length as block_left
        block_left[j] :=  min{k | dist(j, k) < max_dist}.

    '''

    M = len(coords)
    j = 0
    block_left = np.zeros(M)
    for i in xrange(M):
        while j < M and abs(coords[j] - coords[i]) > max_dist:
            j += 1
        block_left[i] = j
    return block_left


def block_left_to_right(block_left):
    '''
    Converts block lefts to block rights.

    Parameters
    ----------
    block_left : array
        Array of block lefts.

    Returns
    -------
    block_right : 1D np.ndarray with same length as block_left
        block_right[j] := max {k | block_left[k] <= j}

    '''
    M = len(block_left)
    j = 0
    block_right = np.zeros(M)
    for i in xrange(M):
        while j < M and block_left[j] <= i:
            j += 1
        block_right[i] = j
    return block_right


def annot_sort_key(s):
    '''For use with --cts-bin. Fixes weird pandas crosstab column order.'''
    if isinstance(s, tuple):
        s = [x.split('_')[0] for x in s]
        s = map(lambda x: float(x) if x != 'min' else -float('inf'), s)
    elif isinstance(s, str):
        s = s.split('_')[0]
        if s == 'min':
            s = float('-inf')
        else:
            s = float(s)
    else:  # bad
        raise ValueError('Something has gone horribly wrong.')
    return s


def cts_bin():
    pass


def read_annot(fh):
    annot = ps.annot_fromlist(fh)
    msg = 'Read %d annotations for %d SNPs from s'
    log.log(msg % (args.annot, n_annot, ma))
    annot.drop(['CM', 'MAF', 'BP'], inplace=True, axis=1)
    if not ps.series_eq(df.SNP, annot.SNP):  # TODO get rid of this requirement
        msg = '--annot must contain same SNPs in same order as --bfile.'
        raise ValueError(msg)
    return annot


def ldscore(args, log):
    '''
    Wrapper function for estimating l1, l1^2, l2 and l4 (+ optionally standard errors) from
    reference panel genotypes.

    Annot format is
    chr snp bp cm <annotations>

    '''
    bim = plink.read_bim(args.bfile+'.bim')
    log.log('Read %d SNP IDs from %s' % (m, snp_file))
    fam = plink.read_fam(args.bfile+'.fam')
    log.log('Read %d individual IDs from %s' % (n, ind_file))
    m, n, len(bim), len(fam)
    if args.annot is not None:
        annot = read_annot(args.annot)
        n_annot, m_annot = len(annot.df.columns)-1, len(annot.df)
    elif args.cts_bin is not None:
        annot = cts_bin(args.cts_bin, args.cts_breaks, args.cts_names)
        n_annot, m_annot = len(annot.df.columns)-1, len(annot.df)
    if args.extract is not None:  # --extract
        keep_snps = __filter__(args.extract, 'SNPs', 'include', array_snps)
        annot_matrix, annot_colnames, n_annot = None, None, 1
    # read fam
    array_indivs = ind_obj(ind_file)
    n = len(array_indivs.IDList)
    if args.keep:
        keep_indivs = __filter__(args.keep, 'individuals', 'include', array_indivs)
    else:
        keep_indivs = None
    covar = ps.read_csv(args.covar, header=0) if args.covar else None
    # read genotypes
    log.log('Reading genotypes from %s' % array_file)
    geno_array = plink.Bfile(array_file, n, array_snps, keep_snps=keep_snps,
                             keep_indivs=keep_indivs, mafMin=args.maf)
    if annot_matrix is not None:  # filter annot to kept SNPs
        annot_matrix = annot_matrix[geno_array.kept_snps, :]
    # determine block widths
    if args.ld_wind_snps:
        max_dist = args.ld_wind_snps
        coords = np.arange(m)
    elif args.ld_wind_kb:
        max_dist = args.ld_wind_kb*1e3
        coords = np.array(array_snps.df['BP'])[geno_array.kept_snps]
    elif args.ld_wind_cm:
        max_dist = args.ld_wind_cm
        coords = np.array(array_snps.df['CM'])[geno_array.kept_snps]
    block_left = ld.getBlockLefts(coords, max_dist)
    if block_left[len(block_left)-1] == 0 and not args.yes_really:
        raise ValueError('Set --yes-really to compute whole-chomosome LD Score.')
    scale_suffix = ''
    if args.pq_exp is not None:
        log.log('Computing LD with pq ^ {S}.'.format(S=args.pq_exp))
        msg = 'Note that LD Scores with pq raised to a nonzero power are'
        msg += 'not directly comparable to normal LD Scores.'
        log.log(msg)
        scale_suffix = '_S{S}'.format(S=args.pq_exp)
        pq = geno_array.maf*(1-geno_array.maf)).reshape((geno_array.m, 1)
        pq = np.power(pq, args.pq_exp)
        if annot_matrix is not None:
            annot_matrix = np.multiply(annot_matrix, pq)
        else:
            annot_matrix = pq
    log.log("Estimating LD Score...")
    lN = geno_array.ldScoreVarBlocks(block_left, args.chunk_size,
                                     annot=annot_matrix, covar=covar)
    col_prefix, file_suffix = "L2", 'L2'
    if n_annot == 1:
        ldscore_colnames = [col_prefix+scale_suffix]
    else:
        ldscore_colnames = [y+col_prefix+scale_suffix for y in annot_colnames]
    df = pd.DataFrame.from_records(np.c_[geno_array.df, lN])
    df.columns = geno_array.colnames + ldscore_colnames
    if args.print_snps:
        c = ps.get_compression(args.print_snps)
        print_snps = ps.read_csv(args.print_snps, header=None, compression=c, usecols=['SNP'])
        msg = 'Read list of %d SNPs for which to print LD Scores from %s'
        log.log(msg % (args.print_snps, len(print_snps)))
        df = df.ix[df.SNP.isin(print_snps.SNP), :]
        if len(df) == 0:
            raise ValueError('After merging with --print-snps, no SNPs remain.')
        else:
            log.log('Printing LD Scores for %d SNPs in --print-snps.' % len(df))
    log.log("Writing LD Scores for %d SNPs to %s.gz" % (out_fname, len(df)))
    df = df.drop(['CM', 'MAF'], axis=1)
    df.to_csv2(args.out+'.l2.ldscore', float_format='%.3f')
    call(['gzip', '-f', out_fname])
    M = np.atleast_1d(np.sum(annot_matrix, axis=0))
    M_5_50 = np.atleast_1d(np.sum(annot_matrix[geno_array.maf > 0.05, :], axis=0))
    print_M(M, args.out+'.l2.M')
    print_M(M_5_50, args.out+'.l2.M_5_50')
    if (args.cts_bin is not None) and not args.noprint_annot:
        print_annot(annot_matrix, args.out)  # TODO this doesn't work yet
    print_ldscore_metadata(df)
    if n_annot > 1:
        print_condnum(df)
    if annot_matrix is not None:
        print_annot_metadata(annot_matrix, log)


def print_M(M, fh):
    '''Print M.'''
    fh = open(fh, 'wb')
    print >>fh, '\t'.join(map(str, M))
    fh.close()


def print_annot(annot_matrix, out):
    '''Print annot matrix produced by --cts-bin.'''
    out += '.annot'
    new_colnames = geno_array.colnames + ldscore_colnames
    annot_df = pd.DataFrame(np.c_[geno_array.df, annot_matrix])
    annot_df.columns = new_colnames
    log.log("Writing annot matrix from --cts-bin to %s" % out+'.gz')
    annot_df.to_csv2(out)
    call(['gzip', '-f', out_fname_annot])


def print_ldscore_metadata(df):
    '''Log LD Score metadata.'''
    pd.set_option('display.max_rows', 200)
    log.log('\nSummary of LD Scores in %s' % out_fname+l2_suffix)
    t = df.ix[:, 4:].describe()
    log.log(t.ix[1:, :]+'\n')
    log.log('MAF/LD Score Correlation Matrix')
    log.log(df.ix[:, 4:].corr())


def print_condnum(df):
    '''Log LD Score matrix condition number.'''
    log.log('\nLD Score Matrix Condition Number')
    cond_num = np.linalg.cond(df.ix[:, 5:])
    log.log(reg.remove_brackets(str(np.matrix(cond_num))))
    if cond_num > 10000:
        log.log('WARNING: ill-conditioned LD Score Matrix!')


def print_annot_metadata(annot_matrix, log):
    '''Log annot matrix metadata to.'''
    x = pd.DataFrame(annot_matrix, columns=annot_colnames)
    log.log('\nAnnotation Correlation Matrix')
    log.log(x.corr())
    log.log('\nAnnotation Matrix Column Sums')
    log.log(_remove_dtype(x.sum(axis=0)))
    log.log('\nSummary of Annotation Matrix Row Sums')
    row_sums = x.sum(axis=1).describe()
    log.log(_remove_dtype(row_sums))
