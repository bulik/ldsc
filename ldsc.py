'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

LDSC is a command line tool for estimating
    1. LD Score
    2. heritability / partitioned heritability
    3. genetic covariance / correlation

'''
from __future__ import division
import ldscore.ldscore as ldscore
import ldscore.sumstats as sumstats
import numpy as np
import pandas as pd
import time
import sys
import traceback
import argparse

__version__ = '2.0'
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
pd.set_option('max_colwidth', 1000)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=4)
try:
    x = pd.DataFrame({'A': [1, 2, 3]})
    x.drop_duplicates(subset='A')
except TypeError:
    raise ImportError('ldsc requires pandas version > 0.15.2')


def get_header(args, MASTHEAD):
    opts = vars(args)
    opts = {x: ' '.join([str(i) for i in opts[x]])
            if type(opts[x]) is list
            else opts[x] for x in
            filter(lambda y: opts[y] is not None, opts)}
    header = MASTHEAD
    header += "\nOptions: \n"
    options = [
        '  --' + x.replace('_', '-') + ' ' + str(opts[x]) for x in opts]
    header += '\n'.join(options) + '\n'
    return header


class Logger(object):
    '''Print to log file and stdout.'''
    def __init__(self, fh):
        self.log_fh = open(fh, 'wb', 0)

    def log(self, msg, stdout=True):
        '''Print to log file and stdout.'''
        print >>self.log_fh, msg
        if stdout:
            x = str(msg).split('\n')
            if len(x) > 20:
                msg = '\n'.join(x[1:10])
                msg += '\nOutput truncated. See log file for full list.'
            sys.stdout.write(str(msg) + '\n')
            sys.stdout.flush()

    def close(self):
        self.log_fh.close()


class ThrowingArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        raise ValueError(message)

    def print_help(self, masthead=True):
        if masthead:
            print MASTHEAD
        argparse.ArgumentParser.print_help(self)

    def which(self):
        try:
            args.bfile
            which = 'ld'
        except AttributeError:
            which = 'reg'
        return which

    def parse_args(self):
        try:
            super(ThrowingArgumentParser, self).parse_args()
        except ValueError:
            print MASTHEAD
            ex_type, ex, tb = sys.exc_info()
            print traceback.format_exc(ex)
            parser.print_help(masthead=False)
            sys.exit(2)


parser = ThrowingArgumentParser()
subparsers = parser.add_subparsers()
# LD Score estimation Flags
ld = subparsers.add_parser('ld', help='Estimate LD Score.')
ld.add_argument('--out', default='ldsc', type=str,
                help='Output filename prefix.')
ld.add_argument('--bfile', default=None, type=str, required=True,
                help='Prefix for Plink .bed/.bim/.fam file')
# Filtering / Data Management for LD Score
annot = ld.add_mutually_exclusive_group(required=False)
annot.add_argument('--extract', default=None, type=str,
                   help='File with SNPs to include in LD Score estimation. '
                   'The file should contain one SNP ID per row.')
annot.add_argument('--annot', default=None, type=str, nargs='+',
                   help='Filename prefix for annotation file for partitioned LD Score estimation. '
                   'LDSC will automatically append .annot or .annot.gz to the filename prefix. '
                   'See docs/file_formats_ld for a definition of the .annot format.')
annot.add_argument('--cts-bin', default=None, type=str,
                   help='This flag tells LDSC to compute partitioned LD Scores, where the partition '
                   'is defined by cutting one or several continuous variable[s] into bins. '
                   'The argument to this flag should be the name of a single file or a comma-separated '
                   'list of files. The file format is two columns, with SNP IDs in the first column '
                   'and the continuous variable in the second column. ')
ld.add_argument('--cts-breaks', default=None, type=str,
                help='Use this flag to specify names for the continuous variables cut into bins '
                'with --cts-bin. For each continuous variable, specify breaks as a comma-separated '
                'list of breakpoints, and separate the breakpoints for each variable with an x. '
                'For example, if binning on MAF and distance to gene (in kb), '
                'you might set --cts-breaks 0.1,0.25,0.4x10,100,1000 ')
ld.add_argument('--keep', default=None, type=str,
                help='File with individuals to include in LD Score estimation. '
                'The file should contain one individual ID per row.')
ld.add_argument('--maf', default=None, type=float,
                help='Minor allele frequency lower bound. Default is MAF > 0.')
ld_wind = ld.add_mutually_exclusive_group(required=True)
ld_wind.add_argument('--ld-wind-snps', default=None, type=int,
                     help='Specify the window size to be used for estimating LD Scores in units of '
                     '# of SNPs. You can only specify one --ld-wind-* option.')
ld_wind.add_argument('--ld-wind-kb', default=None, type=float,
                     help='Specify the window size to be used for estimating LD Scores in units of '
                     'kilobase-pairs (kb). You can only specify one --ld-wind-* option.')
ld_wind.add_argument('--ld-wind-cm', default=None, type=float,
                     help='Specify the window size to be used for estimating LD Scores in units of '
                     'centiMorgans (cM). You can only specify one --ld-wind-* option.')
ld.add_argument('--print-snps', default=None, type=str,
                help='This flag tells LDSC to only print LD Scores for the SNPs listed '
                '(one ID per row) in PRINT_SNPS. The sum r^2 will still include SNPs not in '
                'PRINT_SNPs. This is useful for reducing the number of LD Scores that have to be '
                'read into memory when estimating h2 or rg.')
# Fancy LD Score Estimation Flags
alpha = ld.add_mutually_exclusive_group(required=False)
alpha.add_argument('--per-allele', default=False, action='store_true',
                   help='Setting this flag causes LDSC to compute per-allele LD Scores, '
                   'i.e., \ell_j := \sum_k p_k(1-p_k)r^2_{jk}, where p_k denotes the MAF '
                   'of SNP j. ')
alpha.add_argument('--pq-exp', default=None, type=float,
                   help='Setting this flag causes LDSC to compute LD Scores with the given scale factor, '
                   'i.e., \ell_j := \sum_k (p_k(1-p_k))^a r^2_{jk}, where p_k denotes the MAF '
                   'of SNP j and a is the argument to --pq-exp. ')
ld.add_argument('--no-print-annot', default=False, action='store_true',
                help='By defualt, seting --cts-bin or --cts-bin-add causes LDSC to print '
                'the resulting annot matrix. Setting --no-print-annot tells LDSC not '
                'to print the annot matrix. ')
ld.add_argument('--covar', default=None, type=str,
                help='Covariate file for LD Score estimation (e.g., PCs).')
ld.add_argument('--out', default='ldsc', type=str,
                help='Output filename prefix. If --out is not set, LDSC will use ldsc as the '
                'defualt output filename prefix.')
ld.add_argument('--chunk-size', default=50, type=int,
                help='Chunk size for LD Score calculation. Use the default.')
ld.add_argument('--yes-really', default=False, action='store_true',
                help='Yes, I really want to compute whole-chromosome LD Score.')
# LD Score regression flags
reg = subparsers.add_parser('reg', help='LD Score regression.')
reg.add_argument('--out', default='ldsc', type=str,
                 help='Output filename prefix.')
par = ld.add_mutually_exclusive_group(required=True)
par.add_argument('--h2', default=None, type=str, nargs='+',
                 help='Filename prefix for a .chisq file for one-phenotype LD Score regression. '
                 'LDSC will automatically append .chisq or .chisq.gz to the filename prefix.'
                 '--h2 requires at minimum also setting the --ref-ld and --w-ld flags.')
par.add_argument('--rg', default=None, type=str, nargs='+',
                 help='Comma-separated list of prefixes of .chisq filed for genetic correlation estimation.')
reg.add_argument('--ref-ld', default=None, type=str, required=True, nargs='+',
                 help='Use --ref-ld to tell LDSC which LD Scores to use as the predictors in the LD '
                 'Score regression. '
                 'LDSC will automatically append .l2.ldscore/.l2.ldscore.gz to the filename prefix.')
reg.add_argument('--w-ld', default=None, type=str, required=True,
                 help='Filename prefix for file with LD Scores with sum r^2 taken over SNPs included '
                 'in the regression. LDSC will automatically append .l2.ldscore/.l2.ldscore.gz.')
reg.add_argument('--overlap-annot', default=False, action='store_true',
                 help='This flag informs LDSC that the partitioned LD Scores were generates using an '
                 'annot matrix with overlapping categories (i.e., not all row sums equal 1), '
                 'and prevents LDSC from displaying output that is meaningless with overlapping categories.')
intercept = ld.add_mutually_exclusive_group(required=False)
intercept.add_argument('--no-intercept', action='store_true',
                       help='If used with --h2, this constrains the LD Score regression intercept to equal '
                       '1. If used with --rg, this constrains the LD Score regression intercepts for the h2 '
                       'estimates to be one and the intercept for the genetic covariance estimate to be zero.')
intercept.add_argument('--intercept-h2', action='store', default=None, nargs='+',
                       help='Intercepts for constrained-intercept single-trait LD Score regression.')
intercept.add_argument('--intercept-gencov', action='store', default=None, nargs='+',
                       help='Intercepts for constrained-intercept cross-trait LD Score regression. '
                       'Must have same length as --rg. The first entry is ignored.')
intercept.add_argument('--two-step', default=None, type=float,
                       help='Test statistic bound for use with the two-step estimator.')
reg.add_argument('--M', default=None, type=str, nargs='+',
                 help='# of SNPs (if you don\'t want to use the .l2.M files that came with your .l2.ldscore.gz files)')
reg.add_argument('--chisq-max', default=None, type=float,
                 help='Max chi^2.')
reg.add_argument('--print-cov', default=False, action='store_true',
                 help='For use with --h2/--rg. This flag tells LDSC to print the '
                 'covariance matrix of the estimates.')
reg.add_argument('--print-delete-vals', default=False, action='store_true',
                 help='If this flag is set, LDSC will print the block jackknife delete-values ('
                 'i.e., the regression coefficeints estimated from the data with a block removed). '
                 'The delete-values are formatted as a matrix with (# of jackknife blocks) rows and '
                 '(# of LD Scores) columns.')
reg.add_argument('--n-blocks', default=200, type=int,
                 help='Number of block jackknife blocks.')
reg.add_argument('--not-M-5-50', default=False, action='store_true',
                 help='This flag tells LDSC to use the .l2.M file instead of the .l2.M_5_50 file.')
reg.add_argument('--return-silly-things', default=False, action='store_true',
                 help='Force ldsc to return silly genetic correlation estimates.')
reg.add_argument('--no-check-alleles', default=False, action='store_true',
                 help='For rg estimation, skip checking whether the alleles match. This check is '
                 'redundant for pairs of chisq files generated using munge_sumstats.py and the '
                 'same argument to the --merge-alleles flag.')
reg.add_argument('--print-coefficients', default=False, action='store_true',
                 help='when categories are overlapping, print coefficients as well as heritabilities.')
reg.add_argument('--frqfile', type=str, nargs='+',
                 help='For use with --overlap-annot. Provides allele frequencies to prune to common '
                 'snps if --not-M-5-50 is not set.')
reg.add_argument('--samp-prev', default=None, nargs='+',
                 help='Sample prevalence of binary phenotype (for conversion to liability scale).')
reg.add_argument('--pop-prev', default=None, nargs='+',
                 help='Population prevalence of binary phenotype (for conversion to liability scale).')


if __name__ == '__main__':
    args = parser.parse_args()
    args.which = parser.which()
    log = Logger(args.out + '.log')
    try:
        log.log(get_header(args, MASTHEAD))
        log.log('Beginning analysis at %s' % time.ctime())
        if args.which == 'ld':
            if (args.cts_bin is not None) != (args.cts_breaks is not None):
                raise ValueError(
                    'Must set both or neither of --cts-bin and --cts-breaks.')
            if args.per_allele:
                args.pq_exp = 1
            if (not args.overlap_annot or args.not_M_5_50) and args.frqfile is not None:
                    log.log(
                        'The frequency file is unnecessary and is being ignored.')
                    args.frqfile = None
            ldscore.ldscore(args, log)
        elif args.which == 'reg':
            if (args.samp_prev is not None) != (args.pop_prev is not None):
                raise ValueError(
                    'Must set both or neither of --samp-prev and --pop-prev.')
            if args.rg:
                sumstats.estimate_rg(args, log)
            elif args.h2:
                sumstats.estimate_h2(args, log)
    except Exception:
        ex_type, ex, tb = sys.exc_info()
        log.log(traceback.format_exc(ex), stdout=False)
        raise
    finally:
        log.log('Analysis finished at %s' % T=time.ctime())
        time_elapsed = round(time.time() - start_time, 2)
        log.log('Total time elapsed: %s' % sec_to_str(time_elapsed))
        log.close()
