#!/usr/bin/env python
'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

LDSC is a command line tool for estimating
    1. LD Score
    2. heritability / partitioned heritability
    3. genetic covariance / correlation

'''
from __future__ import division
from __future__ import absolute_import
import ldscore.ldscore as ld
import ldscore.parse as ps
import ldscore.sumstats as sumstats
import ldscore.regressions as reg
import numpy as np
import pandas as pd
import logging
from subprocess import call
from itertools import product
from functools import reduce
import time, sys, traceback, argparse


try:
    x = pd.DataFrame({'A': [1, 2, 3]})
    x.sort_values(by='A')
except AttributeError:
    raise ImportError('LDSC requires pandas version >= 0.17.0')

__version__ = '1.0.0'
MASTHEAD = "*********************************************************************\n"
MASTHEAD += "* LD Score Regression (LDSC)\n"
MASTHEAD += "* Version {V}\n".format(V=__version__)
MASTHEAD += "* (C) 2014-2015 Brendan Bulik-Sullivan and Hilary Finucane\n"
MASTHEAD += "* Broad Institute of MIT and Harvard / MIT Department of Mathematics\n"
MASTHEAD += "* GNU General Public License v3\n"
MASTHEAD += "*********************************************************************\n"
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)
pd.set_option('max_colwidth',1000)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=4)


def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f


def _remove_dtype(x):
    '''Removes dtype: float64 and dtype: int64 from pandas printouts'''
    x = str(x)
    x = x.replace('\ndtype: int64', '')
    x = x.replace('\ndtype: float64', '')
    return x


def __filter__(fname, noun, verb, merge_obj):
    merged_list = None
    if fname:
        f = lambda x,n: x.format(noun=noun, verb=verb, fname=fname, num=n)
        x = ps.FilterFile(fname)
        c = 'Read list of {num} {noun} to {verb} from {fname}'
        logging.info(f(c, len(x.IDList)))
        merged_list = merge_obj.loj(x.IDList)
        len_merged_list = len(merged_list)
        if len_merged_list > 0:
            c = 'After merging, {num} {noun} remain'
            logging.info(f(c, len_merged_list))
        else:
            error_msg = 'No {noun} retained for analysis'
            raise ValueError(f(error_msg, 0))

        return merged_list

def annot_sort_key(s):
    '''For use with --cts-bin. Fixes weird pandas crosstab column order.'''
    if type(s) == tuple:
        s = [x.split('_')[0] for x in s]
        s = map(lambda x: float(x) if x != 'min' else -float('inf'), s)
    else:  # type(s) = str:
        s = s.split('_')[0]
        if s == 'min':
            s = float('-inf')
        else:
            s = float(s)

    return s

def ldscore(args):
    '''
    Wrapper function for estimating l1, l1^2, l2 and l4 (+ optionally standard errors) from
    reference panel genotypes.

    Annot format is
    chr snp bp cm <annotations>

    '''

    if args.bfile:
        snp_file, snp_obj = args.bfile+'.bim', ps.PlinkBIMFile
        ind_file, ind_obj = args.bfile+'.fam', ps.PlinkFAMFile
        array_file, array_obj = args.bfile+'.bed', ld.PlinkBEDFile

    # read bim/snp
    array_snps = snp_obj(snp_file)
    m = len(array_snps.IDList)
    logging.info('Read list of {m} SNPs from {f}'.format(m=m, f=snp_file))
    if args.annot is not None:  # read --annot
        try:
            if args.thin_annot: # annot file has only annotations
                annot = ps.ThinAnnotFile(args.annot)
                n_annot, ma = len(annot.df.columns), len(annot.df)
                logging.info("Read {A} annotations for {M} SNPs from {f}".format(f=args.annot,
                    A=n_annot, M=ma))
                annot_matrix = annot.df.values
                annot_colnames = annot.df.columns
                keep_snps = None
            else:
                annot = ps.AnnotFile(args.annot)
                n_annot, ma = len(annot.df.columns) - 4, len(annot.df)
                logging.info("Read {A} annotations for {M} SNPs from {f}".format(f=args.annot,
                    A=n_annot, M=ma))
                annot_matrix = np.array(annot.df.iloc[:,4:])
                annot_colnames = annot.df.columns[4:]
                keep_snps = None
                if np.any(annot.df.SNP.values != array_snps.df.SNP.values):
                    raise ValueError('The .annot file must contain the same SNPs in the same'+\
                        ' order as the .bim file.')
        except Exception:
            logging.info('Error parsing .annot file')
            raise

    elif args.ld_extract is not None:  # --ld-extract
        keep_snps = __filter__(args.ld_extract, 'SNPs', 'include', array_snps)
        annot_matrix, annot_colnames, n_annot = None, None, 1


    elif args.cts_bin is not None and args.cts_breaks is not None:  # --cts-bin
        cts_fnames = sumstats._splitp(args.cts_bin)  # read filenames
        args.cts_breaks = args.cts_breaks.replace('N','-')  # replace N with negative sign
        try:  # split on x
            breaks = [[float(x) for x in y.split(',')] for y in args.cts_breaks.split('x')]
        except ValueError as e:
            raise ValueError('--cts-breaks must be a comma-separated list of numbers: '
                +str(e.args))

        if len(breaks) != len(cts_fnames):
            raise ValueError('Need to specify one set of breaks for each file in --cts-bin.')

        if args.cts_names:
            cts_colnames = [str(x) for x in args.cts_names.split(',')]
            if len(cts_colnames) != len(cts_fnames):
                msg = 'Must specify either no --cts-names or one value for each file in --cts-bin.'
                raise ValueError(msg)

        else:
            cts_colnames = ['ANNOT'+str(i) for i in range(len(cts_fnames))]

        logging.info('Reading numbers with which to bin SNPs from {F}'.format(F=args.cts_bin))

        cts_levs = []
        full_labs = []
        for i,fh in enumerate(cts_fnames):
            vec = ps.read_cts(cts_fnames[i], array_snps.df.SNP.values)

            max_cts = np.max(vec)
            min_cts = np.min(vec)
            cut_breaks = list(breaks[i])
            name_breaks = list(cut_breaks)
            if np.all(cut_breaks >= max_cts) or np.all(cut_breaks <= min_cts):
                raise ValueError('All breaks lie outside the range of the cts variable.')

            if np.all(cut_breaks <= max_cts):
                name_breaks.append(max_cts)
                cut_breaks.append(max_cts+1)

            if np.all(cut_breaks >= min_cts):
                name_breaks.append(min_cts)
                cut_breaks.append(min_cts-1)

            name_breaks.sort()
            cut_breaks.sort()
            n_breaks = len(cut_breaks)
            # so that col names are consistent across chromosomes with different max vals
            name_breaks[0] = 'min'
            name_breaks[-1] = 'max'
            name_breaks = [str(x) for x in name_breaks]
            labs = [name_breaks[i]+'_'+name_breaks[i+1] for i in range(n_breaks-1)]
            cut_vec = pd.Series(pd.cut(vec, bins=cut_breaks, labels=labs))
            cts_levs.append(cut_vec)
            full_labs.append(labs)

        annot_matrix = pd.concat(cts_levs, axis=1)
        annot_matrix.columns = cts_colnames
        # crosstab -- for now we keep empty columns
        annot_matrix = pd.crosstab(annot_matrix.index,
            [annot_matrix[i] for i in annot_matrix.columns], dropna=False,
            colnames=annot_matrix.columns)

        # add missing columns
        if len(cts_colnames) > 1:
            for x in product(*full_labs):
                if x not in annot_matrix.columns:
                    annot_matrix[x] = 0
        else:
            for x in full_labs[0]:
                if x not in annot_matrix.columns:
                    annot_matrix[x] = 0

        annot_matrix = annot_matrix[sorted(annot_matrix.columns, key=annot_sort_key)]
        if len(cts_colnames) > 1:
            # flatten multi-index
            annot_colnames = ['_'.join([cts_colnames[i]+'_'+b for i,b in enumerate(c)])
                for c in annot_matrix.columns]
        else:
            annot_colnames = [cts_colnames[0]+'_'+b for b in annot_matrix.columns]

        annot_matrix = np.matrix(annot_matrix)
        keep_snps = None
        n_annot = len(annot_colnames)
        if np.any(np.sum(annot_matrix, axis=1) == 0):
            # This exception should never be raised. For debugging only.
            raise ValueError('Some SNPs have no annotation in --cts-bin. This is a bug!')

    else:
        annot_matrix, annot_colnames, keep_snps = None, None, None,
        n_annot = 1

    # read fam
    array_indivs = ind_obj(ind_file)
    n = len(array_indivs.IDList)
    logging.info('Read list of {n} individuals from {f}'.format(n=n, f=ind_file))
    # read keep_indivs
    if args.ld_keep:
        keep_indivs = __filter__(args.ld_keep, 'individuals', 'include', array_indivs)
    else:
        keep_indivs = None

    # read genotype array
    logging.info('Reading genotypes from {fname}'.format(fname=array_file))
    geno_array = array_obj(array_file, n, array_snps, keep_snps=keep_snps,
        keep_indivs=keep_indivs, mafMin=args.maf)

    # filter annot_matrix down to only SNPs passing MAF cutoffs
    if annot_matrix is not None:
        annot_keep = geno_array.kept_snps
        annot_matrix = annot_matrix[annot_keep,:]

    # determine block widths
    x = np.array((args.ld_wind_snps, args.ld_wind_kb, args.ld_wind_cm), dtype=bool)
    if np.sum(x) != 1:
        raise ValueError('Must specify exactly one --ld-wind option')

    if args.ld_wind_snps:
        max_dist = args.ld_wind_snps
        coords = np.array(range(geno_array.m))
    elif args.ld_wind_kb:
        max_dist = args.ld_wind_kb*1000
        coords = np.array(array_snps.df['BP'])[geno_array.kept_snps]
    elif args.ld_wind_cm:
        max_dist = args.ld_wind_cm
        coords = np.array(array_snps.df['CM'])[geno_array.kept_snps]

    block_left = ld.getBlockLefts(coords, max_dist)
    if block_left[len(block_left)-1] == 0 and not args.yes_really:
        error_msg = 'Do you really want to compute whole-chomosome LD Score? If so, set the '
        error_msg += '--yes-really flag (warning: it will use a lot of time / memory)'
        raise ValueError(error_msg)

    scale_suffix = ''
    if args.pq_exp is not None:
        logging.info('Computing LD with pq ^ {S}.'.format(S=args.pq_exp))
        msg = 'Note that LD Scores with pq raised to a nonzero power are'
        msg += 'not directly comparable to normal LD Scores.'
        logging.info(msg)
        scale_suffix = '_S{S}'.format(S=args.pq_exp)
        pq = np.matrix(geno_array.maf*(1-geno_array.maf)).reshape((geno_array.m, 1))
        pq = np.power(pq, args.pq_exp)

        if annot_matrix is not None:
            annot_matrix = np.multiply(annot_matrix, pq)
        else:
            annot_matrix = pq

    logging.info("Estimating LD Score.")
    lN = geno_array.ldScoreVarBlocks(block_left, args.chunk_size, annot=annot_matrix)
    col_prefix = "L2"; file_suffix = "l2"

    if n_annot == 1:
        ldscore_colnames = [col_prefix+scale_suffix]
    else:
        ldscore_colnames =  [y+col_prefix+scale_suffix for y in annot_colnames]

    # print .ldscore. Output columns: CHR, BP, RS, [LD Scores]
    out_fname = args.out + '.' + file_suffix + '.ldscore'
    new_colnames = geno_array.colnames + ldscore_colnames
    df = pd.DataFrame.from_records(np.c_[geno_array.df, lN])
    df.columns = new_colnames
    if args.print_snps:
        if args.print_snps.endswith('gz'):
            print_snps = pd.read_csv(args.print_snps, header=None, compression='gzip')
        elif args.print_snps.endswith('bz2'):
            print_snps = pd.read_csv(args.print_snps, header=None, compression='bz2')
        else:
            print_snps = pd.read_csv(args.print_snps, header=None)
        if len(print_snps.columns) > 1:
            raise ValueError('--print-snps must refer to a file with a one column of SNP IDs.')
        logging.info('Reading list of {N} SNPs for which to print LD Scores from {F}'.format(\
                        F=args.print_snps, N=len(print_snps)))

        print_snps.columns=['SNP']
        df = df.ix[df.SNP.isin(print_snps.SNP),:]
        if len(df) == 0:
            raise ValueError('After merging with --print-snps, no SNPs remain.')
        else:
            msg = 'After merging with --print-snps, LD Scores for {N} SNPs will be printed.'
            logging.info(msg.format(N=len(df)))

    l2_suffix = '.gz'
    logging.info("Writing LD Scores for {N} SNPs to {f}.gz".format(f=out_fname, N=len(df)))
    df.drop(['CM','MAF'], axis=1).to_csv(out_fname, sep="\t", header=True, index=False,
        float_format='%.3f')
    call(['gzip', '-f', out_fname])
    if annot_matrix is not None:
        M = np.atleast_1d(np.squeeze(np.asarray(np.sum(annot_matrix, axis=0))))
        ii = geno_array.maf > 0.05
        M_5_50 = np.atleast_1d(np.squeeze(np.asarray(np.sum(annot_matrix[ii,:], axis=0))))
    else:
        M = [geno_array.m]
        M_5_50 = [np.sum(geno_array.maf > 0.05)]

    # print .M
    fout_M = open(args.out + '.'+ file_suffix +'.M','wb')
    logging.info('\t'.join(map(str,M)), file=fout_M)
    fout_M.close()

    # print .M_5_50
    fout_M_5_50 = open(args.out + '.'+ file_suffix +'.M_5_50','wb')
    logging.info('\t'.join(map(str,M_5_50)), file=fout_M_5_50)
    fout_M_5_50.close()

    # print annot matrix
    if (args.cts_bin is not None) and not args.no_print_annot:
        out_fname_annot = args.out + '.annot'
        new_colnames = geno_array.colnames + ldscore_colnames
        annot_df = pd.DataFrame(np.c_[geno_array.df, annot_matrix])
        annot_df.columns = new_colnames
        del annot_df['MAF']
        logging.info("Writing annot matrix produced by --cts-bin to {F}".format(F=out_fname+'.gz'))
        annot_df.to_csv(out_fname_annot, sep="\t", header=True, index=False)
        call(['gzip', '-f', out_fname_annot])

    # print LD Score summary
    pd.set_option('display.max_rows', 200)
    logging.info('\nSummary of LD Scores in {F}'.format(F=out_fname+l2_suffix))
    t = df.ix[:,4:].describe()
    logging.info( t.ix[1:,:] )

    np.seterr(divide='ignore', invalid='ignore')  # print NaN instead of weird errors
    # print correlation matrix including all LD Scores and sample MAF
    logging.info('')
    logging.info('MAF/LD Score Correlation Matrix')
    logging.info( df.ix[:,4:].corr() )

    # print condition number
    if n_annot > 1: # condition number of a column vector w/ nonzero var is trivially one
        logging.info('\nLD Score Matrix Condition Number')
        cond_num = np.linalg.cond(df.ix[:,5:])
        logging.info( reg.remove_brackets(str(np.matrix(cond_num))) )
        if cond_num > 10000:
            logging.info('WARNING: ill-conditioned LD Score Matrix!')

    # summarize annot matrix if there is one
    if annot_matrix is not None:
        # covariance matrix
        x = pd.DataFrame(annot_matrix, columns=annot_colnames)
        logging.info('\nAnnotation Correlation Matrix')
        logging.info( x.corr() )

        # column sums
        logging.info('\nAnnotation Matrix Column Sums')
        logging.info(_remove_dtype(x.sum(axis=0)))

        # row sums
        logging.info('\nSummary of Annotation Matrix Row Sums')
        row_sums = x.sum(axis=1).describe()
        logging.info(_remove_dtype(row_sums))

    np.seterr(divide='raise', invalid='raise')


parser = argparse.ArgumentParser()

## output file paths
ofile = parser.add_argument_group(title="Output Options", description="Output directory and options to write to files.")
ofile.add_argument('--out', default='ldsc', type=str,
    help='Output filename prefix. If --out is not set, LDSC will use ldsc as the '
    'default output filename prefix.')
ofile.add_argument('--write-excel',  default=False, action='store_true',
    help='This flag tells LDSC to write the estimates from variance components analyses to an excel file. By default, all results included in the log file will be stored in different tabs of the excel file. Additionally, outputs from options such as --print-cov and --print-delete-vals will also be stored as separate tabs of the same file.')
ofile.add_argument('--print-estimates', default=False, action='store_true',
    help='This flag tells LDSC to write the matrices of estimate results to text files.')
ofile.add_argument('--print-cov', default=False, action='store_true',
    help='For use with --h2/--rg. This flag tells LDSC to output the '
    'covaraince matrix of the estimates.')
ofile.add_argument('--print-delete-vals', default=False, action='store_true',
    help='If this flag is set, LDSC will output the block jackknife delete-values ('
    'i.e., the regression coefficeints estimated from the data with a block removed). '
    'The delete-values are formatted as a matrix with (# of jackknife blocks) rows and '
    '(# of LD Scores) columns.')
ofile.add_argument('--stdout-off', default=False, action='store_true',
                    help='Only prints to the log file (not to console).')

# Basic LD Score Estimation Flags
basic_ldsc = parser.add_argument_group(title="LD Score Estimation Flags", description="File paths to Plink format genetic files and l2 scores.")
basic_ldsc.add_argument('--bfile', default=None, type=str,
    help='Prefix for Plink .bed/.bim/.fam file')
basic_ldsc.add_argument('--l2', default=False, action='store_true',
    help='Estimate l2. Compatible with both jackknife and non-jackknife.')

# Filtering / Data Management for LD Score
data_filter = parser.add_argument_group(title="LD Score Estimation Data Filters", description="Data management options for LD scores.")
data_filter.add_argument('--ld-extract', default=None, type=str,
    help='File with SNPs to include in LD Score estimation. '
    'The file should contain one SNP ID per row.')
data_filter.add_argument('--ld-keep', default=None, type=str,
    help='File with individuals to include in LD Score estimation. '
    'The file should contain one individual ID per row.')
data_filter.add_argument('--ld-wind-snps', default=None, type=int,
    help='Specify the window size to be used for estimating LD Scores in units of '
    '# of SNPs. You can only specify one --ld-wind-* option.')
data_filter.add_argument('--ld-wind-kb', default=None, type=float,
    help='Specify the window size to be used for estimating LD Scores in units of '
    'kilobase-pairs (kb). You can only specify one --ld-wind-* option.')
data_filter.add_argument('--ld-wind-cm', default=None, type=float,
    help='Specify the window size to be used for estimating LD Scores in units of '
    'centiMorgans (cM). You can only specify one --ld-wind-* option.')
data_filter.add_argument('--print-snps', default=None, type=str,
    help='This flag tells LDSC to only print LD Scores for the SNPs listed '
    '(one ID per row) in PRINT_SNPS. The sum r^2 will still include SNPs not in '
    'PRINT_SNPs. This is useful for reducing the number of LD Scores that have to be '
    'read into memory when estimating h2 or rg.' )

# Fancy LD Score Estimation Flags
advanced_ldsc = parser.add_argument_group(title="Advanced Options", description="Additional LD score estimation flags")
advanced_ldsc.add_argument('--annot', default=None, type=str,
    help='Filename prefix for annotation file for partitioned LD Score estimation. '
    'LDSC will automatically append .annot or .annot.gz to the filename prefix. '
    'See docs/file_formats_ld for a definition of the .annot format.')
advanced_ldsc.add_argument('--thin-annot', action='store_true', default=False,
    help='This flag says your annot files have only annotations, with no SNP, CM, CHR, BP columns.')
advanced_ldsc.add_argument('--cts-bin', default=None, type=str,
    help='This flag tells LDSC to compute partitioned LD Scores, where the partition '
    'is defined by cutting one or several continuous variable[s] into bins. '
    'The argument to this flag should be the name of a single file or a comma-separated '
    'list of files. The file format is two columns, with SNP IDs in the first column '
    'and the continuous variable in the second column. ')
advanced_ldsc.add_argument('--cts-breaks', default=None, type=str,
    help='Use this flag to specify names for the continuous variables cut into bins '
    'with --cts-bin. For each continuous variable, specify breaks as a comma-separated '
    'list of breakpoints, and separate the breakpoints for each variable with an x. '
    'For example, if binning on MAF and distance to gene (in kb), '
    'you might set --cts-breaks 0.1,0.25,0.4x10,100,1000 ')
advanced_ldsc.add_argument('--cts-names', default=None, type=str,
    help='Use this flag to specify names for the continuous variables cut into bins '
    'with --cts-bin. The argument to this flag should be a comma-separated list of '
    'names. For example, if binning on DAF and distance to gene, you might set '
    '--cts-bin DAF,DIST_TO_GENE ')
advanced_ldsc.add_argument('--per-allele', default=False, action='store_true',
    help='Setting this flag causes LDSC to compute per-allele LD Scores, '
    'i.e., \ell_j := \sum_k p_k(1-p_k)r^2_{jk}, where p_k denotes the MAF '
    'of SNP j. ')
advanced_ldsc.add_argument('--pq-exp', default=None, type=float,
    help='Setting this flag causes LDSC to compute LD Scores with the given scale factor, '
    'i.e., \ell_j := \sum_k (p_k(1-p_k))^a r^2_{jk}, where p_k denotes the MAF '
    'of SNP j and a is the argument to --pq-exp. ')
advanced_ldsc.add_argument('--no-print-annot', default=False, action='store_true',
    help='By defualt, seting --cts-bin or --cts-bin-add causes LDSC to print '
    'the resulting annot matrix. Setting --no-print-annot tells LDSC not '
    'to print the annot matrix. ')
advanced_ldsc.add_argument('--maf', default=None, type=float,
    help='Minor allele frequency lower bound. Default is MAF > 0.')

# Basic Flags for Working with Variance Components
var_components = parser.add_argument_group(title="Variance Components Analyses", description="Basic flags for working with variance components.")
var_components.add_argument('--h2', default=None, type=str,
    help='Filename for a .sumstats[.gz] file for one-phenotype LD Score regression. '
    '--h2 requires at minimum also setting the --ref-ld and --w-ld flags.')
var_components.add_argument('--h2-cts', default=None, type=str,
    help='Filename for a .sumstats[.gz] file for cell-type-specific analysis. '
    '--h2-cts requires the --ref-ld-chr, --w-ld, and --ref-ld-chr-cts flags.')
var_components.add_argument('--rg', default=None, type=str,
    help='Comma-separated list of prefixes of .chisq filed for genetic correlation estimation.')
var_components.add_argument('--ref-ld', default=None, type=str,
    help='Use --ref-ld to tell LDSC which LD Scores to use as the predictors in the LD '
    'Score regression. '
    'LDSC will automatically append .l2.ldscore/.l2.ldscore.gz to the filename prefix. '
    'If this option (and --ref-ld-chr) is not specified, then the file passed from --w-ld is used, '
    'except when calculating partitioned heritability, for which both files must be specified.')
var_components.add_argument('--ref-ld-chr', default=None, type=str,
    help='Same as --ref-ld, but will automatically concatenate .l2.ldscore files split '
    'across 22 chromosomes. LDSC will automatically append .l2.ldscore/.l2.ldscore.gz '
    'to the filename prefix. If the filename prefix contains the symbol @, LDSC will '
    'replace the @ symbol with chromosome numbers. Otherwise, LDSC will append chromosome '
    'numbers to the end of the filename prefix. '
    'Example 1: --ref-ld-chr ld/ will read ld/1.l2.ldscore.gz ... ld/22.l2.ldscore.gz; '
    'Example 2: --ref-ld-chr ld/@_kg will read ld/1_kg.l2.ldscore.gz ... ld/22_kg.l2.ldscore.gz. '
    'If this option (and --ref-ld) is not specified, then the file passed from --w-ld-chr is used, '
    'except when calculating partitioned heritability, for which both files must be specified.')
var_components.add_argument('--ref-ld-chr-cts', default=None, type=str,
    help='Name of a file that has a list of file name prefixes for cell-type-specific analysis.')
var_components.add_argument('--w-ld', default=None, type=str,
    help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included '
    'in the regression. LDSC will automatically append .l2.ldscore/.l2.ldscore.gz. '
    'If this option (and --w-ld-chr) is not specified, then the file passed from --ref-ld is used, '
    'except when calculating partitioned heritability, for which both files must be specified.')
var_components.add_argument('--w-ld-chr', default=None, type=str,
    help='Same as --w-ld, but will read files split into 22 chromosomes in the same '
    'manner as --ref-ld-chr. '
    'If this option (and --w-ld) is not specified, then the file passed from --ref-ld-chr is used, '
    'except when calculating partitioned heritability, for which both files must be specified.')

# Filtering / data management for variance component analysis
var_options = parser.add_argument_group(title="Variance Components Advanced Options", description="Advanced Options for variance components analyses.")
var_options.add_argument('--exclude', type=str, default=None, help='A list of SNPs to be excluded from variance components analyses.')
var_options.add_argument('--overlap-annot', default=False, action='store_true',
    help='This flag informs LDSC that the partitioned LD Scores were generates using an '
    'annot matrix with overlapping categories (i.e., not all row sums equal 1), '
    'and prevents LDSC from displaying output that is meaningless with overlapping categories.')
var_options.add_argument('--print-coefficients',default=False,action='store_true',
    help='when categories are overlapping, print coefficients as well as heritabilities.')
var_options.add_argument('--frqfile', type=str,
    help='For use with --overlap-annot. Provides allele frequencies to prune to common '
    'snps if --not-M-5-50 is not set.')
var_options.add_argument('--frqfile-chr', type=str,
    help='Prefix for --frqfile files split over chromosome.')
var_options.add_argument('--no-intercept', action='store_true',
    help = 'If used with --h2, this constrains the LD Score regression intercept to equal '
    '1. If used with --rg, this constrains the LD Score regression intercepts for the h2 '
    'estimates to be one and the intercept for the genetic covariance estimate to be zero.')
var_options.add_argument('--intercept-h2', action='store', default=None,
    help = 'Intercepts for constrained-intercept single-trait LD Score regression.')
var_options.add_argument('--intercept-gencov', action='store', default=None,
    help = 'Intercepts for constrained-intercept cross-trait LD Score regression.'
    ' Must have same length as --rg. The first entry is ignored.')
var_options.add_argument('--M', default=None, type=str,
    help='# of SNPs (if you don\'t want to use the .l2.M files that came with your .l2.ldscore.gz files)')
var_options.add_argument('--two-step', default=None, type=float,
    help='Test statistic bound for use with the two-step estimator. Not compatible with --no-intercept and --constrain-intercept.')
var_options.add_argument('--chisq-max', default=None, type=float,
    help='Max chi^2.')
var_options.add_argument('--print-all-cts', action='store_true', default=False)

# Flags you should almost never use
rare_options = parser.add_argument_group(title="Rare Options", description="These flags are scarcely used.")
rare_options.add_argument('--chunk-size', default=50, type=int,
    help='Chunk size for LD Score calculation. Use the default.')
rare_options.add_argument('--pickle', default=False, action='store_true',
    help='Store .l2.ldscore files as pickles instead of gzipped tab-delimited text.')
rare_options.add_argument('--yes-really', default=False, action='store_true',
    help='Yes, I really want to compute whole-chromosome LD Score.')
rare_options.add_argument('--invert-anyway', default=False, action='store_true',
    help="Force LDSC to attempt to invert ill-conditioned matrices.")
rare_options.add_argument('--n-blocks', default=200, type=int,
    help='Number of block jackknife blocks.')
rare_options.add_argument('--not-M-5-50', default=False, action='store_true',
    help='This flag tells LDSC to use the .l2.M file instead of the .l2.M_5_50 file.')
rare_options.add_argument('--return-silly-things', default=False, action='store_true',
    help='Force ldsc to return silly genetic correlation estimates.')
rare_options.add_argument('--no-check-alleles', default=False, action='store_true',
    help='For rg estimation, skip checking whether the alleles match. This check is '
    'redundant for pairs of chisq files generated using munge_sumstats.py and the '
    'same argument to the --merge-alleles flag.')

# transform to liability scale
liability = parser.add_argument_group(title="Scale Transformation", description="Setting for libaility scale transformation.")
liability.add_argument('--samp-prev',default=None,
    help='Sample prevalence of binary phenotype (for conversion to liability scale).')
liability.add_argument('--pop-prev',default=None,
    help='Population prevalence of binary phenotype (for conversion to liability scale).')

if __name__ == '__main__':

    args = parser.parse_args()
    if args.out is None:
        raise ValueError('--out is required.')

    logging.basicConfig(format='%(message)s', filename=args.out+'.log', filemode='w', level=logging.INFO)
    if not args.stdout_off: # suppress printing to console
        logging.getLogger().addHandler(logging.StreamHandler())

    try:
        defaults = vars(parser.parse_args(''))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = MASTHEAD
        header += "Call: \n"
        header += './ldsc.py \\\n'
        options = ['--'+x.replace('_','-')+' '+str(opts[x])+' \\' for x in non_defaults]
        header += '\n'.join(options).replace('True','').replace('False','')
        header = header[0:-1]+'\n'
        logging.info(header)
        logging.info('Beginning analysis at {T}'.format(T=time.ctime()))
        start_time = time.time()
        if args.n_blocks <= 1:
            raise ValueError('--n-blocks must be an integer > 1.')
        if args.bfile is not None:
            if args.l2 is None:
                raise ValueError('Must specify --l2 with --bfile.')
            if args.annot is not None and args.ld_extract is not None:
                raise ValueError('--annot and --ld-extract are currently incompatible.')
            if args.cts_bin is not None and args.ld_extract is not None:
                raise ValueError('--cts-bin and --ld-extract are currently incompatible.')
            if args.annot is not None and args.cts_bin is not None:
                raise ValueError('--annot and --cts-bin are currently incompatible.')
            if (args.cts_bin is not None) != (args.cts_breaks is not None):
                raise ValueError('Must set both or neither of --cts-bin and --cts-breaks.')
            if args.per_allele and args.pq_exp is not None:
                raise ValueError('Cannot set both --per-allele and --pq-exp (--per-allele is equivalent to --pq-exp 1).')
            if args.per_allele:
                args.pq_exp = 1


            ldscore(args)
        # summary statistics
        elif (args.h2 or args.rg or args.h2_cts) and (args.ref_ld or args.ref_ld_chr or args.w_ld or args.w_ld_chr):
            if args.h2 is not None and args.rg is not None:
                raise ValueError('Cannot set both --h2 and --rg.')
            if args.ref_ld and args.ref_ld_chr:
                raise ValueError('Cannot set both --ref-ld and --ref-ld-chr.')
            if args.w_ld and args.w_ld_chr:
                raise ValueError('Cannot set both --w-ld and --w-ld-chr.')
            if args.overlap_annot:
                if not (args.ref_ld or args.ref_ld_chr) or not (args.w_ld or args.w_ld_chr):
                    raise ValueError('For partitioned heritability calculation, both --ref-ld (or --ref-ld-chr) and --w-ld (or --w-ld-chr) should be specified.')
            else:
                if (args.ref_ld or args.ref_ld_chr) and not (args.w_ld or args.w_ld_chr):
                    if args.ref_ld is not None:
                        args.w_ld = args.ref_ld
                        logging.info('Options --w-ld or --w-ld-chr not specified; automatically using --ref-ld for --w-ld.')
                    elif args.ref_ld_chr is not None:
                        args.w_ld_chr = args.ref_ld_chr
                        logging.info('Options --w-ld or --w-ld-chr not specified; automatically using --ref-ld-chr for --w-ld-chr.')
                elif not (args.ref_ld or args.ref_ld_chr) and (args.w_ld or args.w_ld_chr):
                    if args.w_ld is not None:
                        args.ref_ld = args.w_ld
                        logging.info('Options --ref-ld or --ref-ld-chr not specified; automatically using --w-ld for --ref-ld.')
                    elif args.w_ld_chr is not None:
                        args.ref_ld_chr = args.w_ld_chr
                        logging.info('Options --ref-ld or --ref-ld-chr not specified; automatically using --w-ld-chr for --ref-ld-chr.')
            if (args.samp_prev is not None) != (args.pop_prev is not None):
                raise ValueError('Must set both or neither of --samp-prev and --pop-prev.')

            if not args.overlap_annot or args.not_M_5_50:
                if args.frqfile is not None or args.frqfile_chr is not None:
                    logging.info('The frequency file is unnecessary and is being ignored.')
                    args.frqfile = None
                    args.frqfile_chr = None
            if args.overlap_annot and not args.not_M_5_50:
                if not ((args.frqfile and args.ref_ld) or (args.frqfile_chr and args.ref_ld_chr)):
                    raise ValueError('Must set either --frqfile and --ref-ld or --frqfile-chr and --ref-ld-chr')

            if args.rg:
                sumstats.estimate_rg(args)
            elif args.h2:
                sumstats.estimate_h2(args)
            elif args.h2_cts:
                sumstats.cell_type_specific(args)

            # bad flags
        else:
            logging.info(header)
            logging.info('Error: no analysis selected.')
            logging.info('ldsc.py -h describes options.')
    except Exception:
        ex_type, ex, tb = sys.exc_info()
        logging.info( traceback.format_exc(ex) )
        raise
    finally:
        logging.info('Analysis finished at {T}'.format(T=time.ctime()) )
        time_elapsed = round(time.time()-start_time,2)
        logging.info('Total time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))
