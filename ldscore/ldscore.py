from __future__ import division
import numpy as np
import parse as ps
import plink
import itertools as it
import pandas as pd
from subprocess import call


def _remove_dtype(x):
    '''Removes dtype: float64 and dtype: int64 from pandas printouts'''
    x = str(x)
    x = x.replace('\ndtype: int64', '')
    x = x.replace('\ndtype: float64', '')
    return x


def to_csv(fh, **kwargs):
    '''pd.DataFrame.to_csv with some defaults filled in.'''
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


def read_annot(fh, log):
    annot = ps.annot_fromlist(fh)
    annot.drop(['CM', 'MAF', 'BP'], inplace=True, axis=1)
    n_annot, m_annot = annot.shape - (1, 0)
    msg = 'Read %d annotations for %d SNPs from %s.'
    log.log(msg % (n_annot, m_annot, fh))
    return annot


def ldscore(args, log):
    '''Estimate LD Score.'''
    fam = plink.read_fam(args.bfile+'.fam')
    n = len(fam)
    log.log('Read %d individual IDs from %s' % (n, args.bfile+'.fam'))
    bim = plink.read_bim(args.bfile+'.bim')
    m = len(bim)
    log.log('Read %d SNP IDs from %s' % (m, args.bfile+'.bim'))
    if args.annot is not None:
        annot = read_annot(args.annot)
        n_annot = len(annot.df.columns)-1
    elif args.cts_bin is not None:
        annot = ps.cts_dummies(ps.cts_fromlist(args.cts_bin), args.cts_breaks)
        n_annot = len(annot.columns)-1
    if args.extract is not None:
        keep_indivs = None
    if args.keep:
        keep_snps = None
    covar = ps.read_csv(args.covar, header=0) if args.covar else None
    # read genotypes
    log.log('Reading genotypes from %s' % args.bfile+'.bed')
    geno = plink.Bfile(args.bfile+'.bed', n, bim, keep_snps=keep_snps,
                             keep_indivs=keep_indivs, mafMin=args.maf)
    # kill singleton SNPs from annot
    pass
    # determine block widths
    if args.ld_wind_snps:
        max_dist = args.ld_wind_snps
        coords = np.arange(m)
    elif args.ld_wind_kb:
        max_dist = args.ld_wind_kb*1e3
        coords = bim.BP
    elif args.ld_wind_cm:
        max_dist = args.ld_wind_cm
        coords = bim.CM
    block_left = block_lefts(coords, max_dist)
    if block_left[len(block_left)-1] == 0 and not args.yes_really:
        raise ValueError('Set --yes-really to compute whole-chomosome LD Score.')
    scale_suffix = ''
    if args.pq_exp is not None:
        log.log('Computing LD with pq ^ {S}.'.format(S=args.pq_exp))
        msg = 'Note that LD Scores with pq raised to a nonzero power are'
        msg += 'not directly comparable to normal LD Scores.'
        log.log(msg)
        scale_suffix = '_S{S}'.format(S=args.pq_exp)
        pq = geno.maf*(1-geno.maf).reshape((geno.m, 1))
        pq = np.power(pq, args.pq_exp)
        if annot is not None:
            annot = np.multiply(annot, pq)
        else:
            annot = pq
    log.log("Estimating LD Score...")
    lN = geno.ld_score(block_left, args.chunk_size, annot=annot, covar=covar)
    if n_annot == 1:
        ldscore_colnames = ['L2'+scale_suffix]
    else:
        ldscore_colnames = [y+'L2'+scale_suffix for y in annot.columns]
    df = pd.DataFrame.from_records(np.c_[geno.df, lN])
    df.columns = geno.colnames + ldscore_colnames
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
    M = np.atleast_1d(np.sum(annot, axis=0))
    M_5_50 = np.atleast_1d(np.sum(annot[geno.maf > 0.05, :], axis=0))
    print_ldscore(df, args.out+'.l2.ldscore', log)
    print_M(M, args.out+'.l2.M')
    print_M(M_5_50, args.out+'.l2.M_5_50')
    if (args.cts_bin is not None) and not args.noprint_annot:
        print_annot(annot, args.out, log)  # TODO this doesn't work yet
    print_ldscore_metadata(df, log)
    if n_annot > 1:
        print_condnum(df, log)
    if annot is not None:
        print_annot_metadata(annot, log)


def print_ldscore(df, fh, log):
    log.log("Writing LD Scores for %d SNPs to %s.gz" % (fh, len(df)))
    df = df.drop(['CM', 'MAF'], axis=1)
    df.to_csv2(fh, float_format='%.3f')
    call(['gzip', '-f', fh])


def print_M(M, fh):
    fh = open(fh, 'wb')
    print >>fh, '\t'.join(map(str, M))
    fh.close()


def print_annot(annot, fh, log):
    log.log("Writing annot matrix from --cts-bin to %s" % fh+'.gz')
    annot.to_csv2(fh)
    call(['gzip', '-f', fh])


def print_ldscore_metadata(df, fh, log):
    pd.set_option('display.max_rows', 200)
    log.log('\nSummary of LD Scores in %s.gz' % fh)
    t = df.ix[:, 4:].describe()
    log.log(t.ix[1:, :]+'\n')
    log.log('MAF/LD Score Correlation Matrix')
    log.log(df.ix[:, 4:].corr())


def print_condnum(df, log):
    log.log('\nLD Score Matrix Condition Number')
    cond_num = np.linalg.cond(df.ix[:, 5:])
    log.log(round(cond_num, 0))
    if cond_num > 10000:
        log.log('WARNING: ill-conditioned LD Score Matrix!')


def print_annot_metadata(annot, log):
    log.log('\nAnnotation Correlation Matrix')
    log.log(annot.corr())
    log.log('\nAnnotation Matrix Column Sums')
    log.log(_remove_dtype(annot.sum(axis=0)))
    log.log('\nSummary of Annotation Matrix Row Sums')
    row_sums = annot.sum(axis=1).describe()
    log.log(_remove_dtype(row_sums))
