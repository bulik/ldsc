'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This module deals with getting all the data needed for LD Score regression from files
into memory and checking that the input makes sense. There is no math here. LD Score
regression is implemented in the regressions module.

'''
from __future__ import division
from __future__ import absolute_import
import numpy as np
import pandas as pd
from scipy import stats
import itertools as it
import ldscore.parse as ps
import ldscore.regressions as reg
import sys
import traceback
import logging
import copy
import os


_N_CHR = 22
# complementary bases
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
# bases
BASES = COMPLEMENT.keys()
# true iff strand ambiguous
STRAND_AMBIGUOUS = {''.join(x): x[0] == COMPLEMENT[x[1]]
                    for x in it.product(BASES, BASES)
                    if x[0] != x[1]}
# SNPS we want to keep (pairs of alleles)
VALID_SNPS = {x for x in map(lambda y: ''.join(y), it.product(BASES, BASES))
              if x[0] != x[1] and not STRAND_AMBIGUOUS[x]}
# T iff SNP 1 has the same alleles as SNP 2 (allowing for strand or ref allele flip).
MATCH_ALLELES = {x for x in map(lambda y: ''.join(y), it.product(VALID_SNPS, VALID_SNPS))
                 # strand and ref match
                 if ((x[0] == x[2]) and (x[1] == x[3])) or
                 # ref match, strand flip
                 ((x[0] == COMPLEMENT[x[2]]) and (x[1] == COMPLEMENT[x[3]])) or
                 # ref flip, strand match
                 ((x[0] == x[3]) and (x[1] == x[2])) or
                 ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))}  # strand and ref flip
# T iff SNP 1 has the same alleles as SNP 2 w/ ref allele flip.
FLIP_ALLELES = {''.join(x):
                ((x[0] == x[3]) and (x[1] == x[2])) or  # strand match
                # strand flip
                ((x[0] == COMPLEMENT[x[3]]) and (x[1] == COMPLEMENT[x[2]]))
                for x in MATCH_ALLELES}


def _splitp(fstr):
    flist = fstr.split(',')
    flist = [os.path.expanduser(os.path.expandvars(x)) for x in flist]
    return flist


def _select_and_log(x, ii, msg):
    '''Fiter down to rows that are True in ii. Log # of SNPs removed.'''
    new_len = ii.sum()
    if new_len == 0:
        raise ValueError(msg.format(N=0))
    else:
        x = x[ii]
        logging.info(msg.format(N=new_len))
    return x


def smart_merge(x, y):
    '''Check if SNP columns are equal. If so, save time by using concat instead of merge.'''
    if len(x) == len(y) and (x.index == y.index).all() and (x.SNP == y.SNP).all():
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True).drop('SNP', 1)
        out = pd.concat([x, y], axis=1)
    else:
        out = pd.merge(x, y, how='inner', on='SNP')
    return out


def _read_ref_ld(args):
    '''Read reference LD Scores.'''
    ref_ld = _read_chr_split_files(args.ref_ld_chr, args.ref_ld,
                                   'reference panel LD Score', ps.ldscore_fromlist)
    logging.info(
        'Read reference panel LD Scores for {N} SNPs.'.format(N=len(ref_ld)))
    return ref_ld


def _read_annot(args):
    '''Read annot matrix.'''
    try:
        if args.ref_ld is not None:
            overlap_matrix, M_tot = _read_chr_split_files(args.ref_ld_chr, args.ref_ld,
                                                          'annot matrix', ps.annot, frqfile=args.frqfile)
        elif args.ref_ld_chr is not None:
            overlap_matrix, M_tot = _read_chr_split_files(args.ref_ld_chr, args.ref_ld,
                                                      'annot matrix', ps.annot, frqfile=args.frqfile_chr)
    except Exception:
        logging.info('Error parsing .annot file.')
        raise

    return overlap_matrix, M_tot


def _read_M(args, n_annot):
    '''Read M (--M, --M-file, etc).'''
    if args.M:
        try:
            M_annot = [float(x) for x in _splitp(args.M)]
        except ValueError as e:
            raise ValueError('Could not cast --M to float: ' + str(e.args))
    else:
        if args.ref_ld:
            M_annot = ps.M_fromlist(
                _splitp(args.ref_ld), common=(not args.not_M_5_50))
        elif args.ref_ld_chr:
            M_annot = ps.M_fromlist(
                _splitp(args.ref_ld_chr), _N_CHR, common=(not args.not_M_5_50))

    try:
        M_annot = np.array(M_annot).reshape((1, n_annot))
    except ValueError as e:
        raise ValueError(
            '# terms in --M must match # of LD Scores in --ref-ld.\n' + str(e.args))

    return M_annot


def _read_w_ld(args):
    '''Read regression SNP LD.'''
    if (args.w_ld and ',' in args.w_ld) or (args.w_ld_chr and ',' in args.w_ld_chr):
        raise ValueError(
            '--w-ld must point to a single fileset (no commas allowed).')
    w_ld = _read_chr_split_files(args.w_ld_chr, args.w_ld,
                                 'regression weight LD Score', ps.ldscore_fromlist)
    if len(w_ld.columns) != 2:
        raise ValueError('--w-ld may only have one LD Score column.')
    w_ld.columns = ['SNP', 'LD_weights']  # prevent colname conflicts w/ ref ld
    logging.info(
        'Read regression weight LD Scores for {N} SNPs.'.format(N=len(w_ld)))
    return w_ld


def _read_chr_split_files(chr_arg, not_chr_arg, noun, parsefunc, **kwargs):
    '''Read files split across 22 chromosomes (annot, ref_ld, w_ld).'''
    try:
        if not_chr_arg:
            logging.info('Reading {N} from {F} ...'.format(F=not_chr_arg, N=noun))
            out = parsefunc(_splitp(not_chr_arg), **kwargs)
        elif chr_arg:
            f = ps.sub_chr(chr_arg, '[1-22]')
            logging.info('Reading {N} from {F} ...'.format(F=f, N=noun))
            out = parsefunc(_splitp(chr_arg), _N_CHR, **kwargs)
    except ValueError as e:
        logging.info('Error parsing {N}.'.format(N=noun))
        raise e

    return out


def _read_sumstats(args, fh, alleles=False, dropna=False):
    '''Parse summary statistics.'''
    logging.info('Reading summary statistics from {S} ...'.format(S=fh))
    sumstats = ps.sumstats(fh, alleles=alleles, dropna=dropna)
    log_msg = 'Read summary statistics for {N} SNPs.'
    logging.info(log_msg.format(N=len(sumstats)))
    m = len(sumstats)
    sumstats = sumstats.drop_duplicates(subset='SNP')
    if m > len(sumstats):
        logging.info(
            'Dropped {M} SNPs with duplicated rs numbers.'.format(M=m - len(sumstats)))

    return sumstats


def _check_ld_condnum(args, ref_ld):
    '''Check condition number of LD Score matrix.'''
    if len(ref_ld.shape) >= 2:
        cond_num = int(np.linalg.cond(ref_ld))
        if cond_num > 100000:
            if args.invert_anyway:
                warn = "WARNING: LD Score matrix condition number is {C}. "
                warn += "Inverting anyway because the --invert-anyway flag is set."
                logging.info(warn.format(C=cond_num))
            else:
                warn = "WARNING: LD Score matrix condition number is {C}. "
                warn += "Remove collinear LD Scores. "
                raise ValueError(warn.format(C=cond_num))


def _check_variance(M_annot, ref_ld):
    '''Remove zero-variance LD Scores.'''
    ii = ref_ld.iloc[:, 1:].var() == 0  # NB there is a SNP column here
    if ii.all():
        raise ValueError('All LD Scores have zero variance.')
    else:
        logging.info('Removing partitioned LD Scores with zero variance.')
        ii_snp = np.array([True] + list(~ii))
        ii_m = np.array(~ii)
        ref_ld = ref_ld.iloc[:, ii_snp]
        M_annot = M_annot[:, ii_m]

    return M_annot, ref_ld, ii


def _warn_length(sumstats):
    if len(sumstats) < 200000:
        logging.info(
            'WARNING: number of SNPs less than 200k; this is almost always bad.')


def _print_cov(ldscore_reg, ofh):
    '''Prints covariance matrix of slopes.'''
    logging.info(
        'Printing covariance matrix of the estimates to {F}.'.format(F=ofh))
    np.savetxt(ofh, ldscore_reg.coef_cov)


def _print_delete_values(ldscore_reg, ofh):
    '''Prints block jackknife delete-k values'''
    logging.info('Printing block jackknife delete values to {F}.'.format(F=ofh))
    np.savetxt(ofh, ldscore_reg.tot_delete_values)

def _print_part_delete_values(ldscore_reg, ofh):
    '''Prints partitioned block jackknife delete-k values'''
    logging.info('Printing partitioned block jackknife delete values to {F}.'.format(F=ofh))
    np.savetxt(ofh, ldscore_reg.part_delete_values)


def _merge_and_log(ld, sumstats, noun):
    '''Wrap smart merge with log messages about # of SNPs.'''
    sumstats = smart_merge(ld, sumstats)
    msg = 'After merging with {F}, {N} SNPs remain.'
    if len(sumstats) == 0:
        raise ValueError(msg.format(N=len(sumstats), F=noun))
    else:
        logging.info(msg.format(N=len(sumstats), F=noun))

    return sumstats

def _read_ref(args):

    ref_ld = _read_ref_ld(args)
    n_annot = len(ref_ld.columns) - 1
    M_annot = _read_M(args, n_annot)
    M_annot, ref_ld, novar_cols = _check_variance(M_annot, ref_ld)
    w_ld = _read_w_ld(args)
    ref_ld_cnames = ref_ld.columns[1:len(ref_ld.columns)]

    return ref_ld, w_ld, M_annot, ref_ld_cnames, novar_cols

def _read_ld_sumstats(args, fh, ref_ld, w_ld, alleles=False, dropna=True):
    if type(fh) == pd.DataFrame:
        sumstats=fh.dropna(how='any')
        msg = '{N} variants are dropped due to N/A values.'
        logging.info(msg.format(N=len(fh)-len(sumstats)))
    else:
        sumstats = _read_sumstats(args, fh, alleles=alleles, dropna=dropna)

    sumstats = _merge_and_log(ref_ld, sumstats, 'reference panel LD')
    sumstats = _merge_and_log(sumstats, w_ld, 'regression SNP LD')
    w_ld_cname = sumstats.columns[-1]

    if args.exclude: #--exclude
        df_exclude = pd.read_csv(args.exclude, index_col=None, header=None)
        df_exclude.columns = ['SNP']
        sumstats = sumstats[~sumstats.SNP.isin(df_exclude.SNP)]
        msg = 'After excluding variants from {F}, {N} SNPs remain.'
        if len(sumstats) == 0:
            raise ValueError(msg.format(N=len(sumstats), F='--exclude SNP list'))
        else:
            logging.info(msg.format(N=len(sumstats), F='--exclude SNP list'))

    return sumstats, w_ld_cname

def cell_type_specific(args):
    '''Cell type specific analysis'''
    args = copy.deepcopy(args)
    if args.intercept_h2 is not None:
        args.intercept_h2 = float(args.intercept_h2)
    if args.no_intercept:
        args.intercept_h2 = 1

    ref_ld, w_ld, M_annot_all_regr, ref_ld_cnames,novar_cols = _read_ref(args)
    sumstats, w_ld_cname = _read_ld_sumstats(args, args.h2_cts, ref_ld, w_ld, alleles=True, dropna=True)
    M_tot = np.sum(M_annot_all_regr)
    _check_ld_condnum(args, ref_ld_cnames_all_regr)
    _warn_length(sumstats)
    n_snp = len(sumstats)
    n_blocks = min(n_snp, args.n_blocks)
    if args.chisq_max is None:
        chisq_max = max(0.001*sumstats.N.max(), 80)
    else:
        chisq_max = args.chisq_max

    ii = np.ravel(sumstats.Z**2 < chisq_max)
    sumstats = sumstats.loc[ii, :]
    logging.info('Removed {M} SNPs with chi^2 > {C} ({N} SNPs remain)'.format(
            C=chisq_max, N=np.sum(ii), M=n_snp-np.sum(ii)))
    n_snp = np.sum(ii)  # lambdas are late-binding, so this works
    ref_ld_all_regr = np.array(sumstats[ref_ld_cnames_all_regr]).reshape((len(sumstats),-1))
    chisq = np.array(sumstats.Z**2)
    keep_snps = sumstats[['SNP']]

    s = lambda x: np.array(x).reshape((n_snp, 1))
    results_columns = ['Name', 'Coefficient', 'Coefficient_std_error', 'Coefficient_P_value']
    results_data = []
    for (name, ct_ld_chr) in [x.split() for x in open(args.ref_ld_chr_cts).readlines()]:
        ref_ld_cts_allsnps = _read_chr_split_files(ct_ld_chr, None,
                                   'cts reference panel LD Score', ps.ldscore_fromlist)
        logging.info('Performing regression.')
        ref_ld_cts = np.array(pd.merge(keep_snps, ref_ld_cts_allsnps, on='SNP', how='left').loc[:,1:])
        if np.any(np.isnan(ref_ld_cts)):
            raise ValueError ('Missing some LD scores from cts files. Are you sure all SNPs in ref-ld-chr are also in ref-ld-chr-cts')

        ref_ld = np.hstack([ref_ld_cts, ref_ld_all_regr])
        M_cts = ps.M_fromlist(
                _splitp(ct_ld_chr), _N_CHR, common=(not args.not_M_5_50))
        M_annot = np.hstack([M_cts, M_annot_all_regr])
        hsqhat = reg.Hsq(s(chisq), ref_ld, s(sumstats[w_ld_cname]), s(sumstats.N),
                     M_annot, n_blocks=n_blocks, intercept=args.intercept_h2,
                     twostep=None, old_weights=True)
        coef, coef_se = hsqhat.coef[0], hsqhat.coef_se[0]
        results_data.append((name, coef, coef_se, stats.norm.sf(coef/coef_se)))
        if args.print_all_cts:
            for i in range(1, len(ct_ld_chr.split(','))):
                coef, coef_se = hsqhat.coef[i], hsqhat.coef_se[i]
                results_data.append((name+'_'+str(i), coef, coef_se, stats.norm.sf(coef/coef_se)))


    df_results = pd.DataFrame(data = results_data, columns = results_columns)
    df_results.sort_values(by = 'Coefficient_P_value', inplace=True)
    df_results.to_csv(args.out+'.cell_type_results.txt', sep='\t', index=False)
    logging.info('Results printed to '+args.out+'.cell_type_results.txt')


def estimate_h2(args):
    '''Estimate h2 and partitioned h2.'''
    #args = copy.deepcopy(args)
    if args.samp_prev is not None and args.pop_prev is not None:
        args.samp_prev, args.pop_prev = map(
            float, [args.samp_prev, args.pop_prev])
    if args.intercept_h2 is not None:
        args.intercept_h2 = float(args.intercept_h2)
    if args.no_intercept:
        args.intercept_h2 = 1
    ref_ld_orig, w_ld_orig, M_annot_orig, ref_ld_cnames_orig, novar_cols_orig = _read_ref(args)

    H2 = []
    sumstats_list = {}

    if isinstance(args.h2, str): # command line case
        h2_paths, h2_files = _parse_(args.h2) # rows
        h2_list = h2_paths # columns
        n_pheno = len(h2_paths)
    else: # python case
        h2_frames = args.h2 # rows
        h2_list = list(h2_frames) # columns
        n_pheno = len(args.h2)
        h2_files =  []
        for i in range(n_pheno):
            h2_files.append("Trait_{}".format(i+1))

    if args.two_step is not None:
        logging.info('Using two-step estimator with cutoff at {M}.'.format(M=args.two_step))

    for k, p1_orig in enumerate(h2_list):

        ref_ld, w_ld, M_annot, ref_ld_cnames, novar_cols=map(lambda x: x.copy(),(ref_ld_orig, w_ld_orig, M_annot_orig, ref_ld_cnames_orig, novar_cols_orig))

        if isinstance(args.h2, str):
            p1 = p1_orig
        else:
            p1 = p1_orig.copy()

        sumstats, w_ld_cname = _read_ld_sumstats(args, p1, ref_ld, w_ld, alleles=True, dropna=True)
        sumstats_list[k] = pd.DataFrame(sumstats)
        ref_ld = np.array(sumstats[ref_ld_cnames])
        _check_ld_condnum(args, ref_ld_cnames)
        _warn_length(sumstats)
        n_snp = len(sumstats)
        n_blocks = min(n_snp, args.n_blocks)

        n_annot = len(ref_ld_cnames)
        chisq_max = args.chisq_max
        old_weights = False
        if n_annot == 1:
            if args.two_step is None and args.intercept_h2 is None:
                args.two_step = float('inf')
        else:
            old_weights = True
            if args.chisq_max is None:
                chisq_max = max(0.001*sumstats.N.max(), 80)

        s = lambda x: np.array(x).reshape((n_snp, 1))
        chisq = s(sumstats.Z**2)
        if chisq_max is not None:
            ii = np.ravel(chisq < chisq_max)
            sumstats = sumstats.loc[ii, :]
            logging.info('Removed {M} SNPs with chi^2 > {C} ({N} SNPs remain)'.format(
                    C=chisq_max, N=np.sum(ii), M=n_snp-np.sum(ii)))
            n_snp = np.sum(ii)  # lambdas are late-binding, so this works
            ref_ld = np.array(sumstats[ref_ld_cnames])
            chisq = chisq[ii].reshape((n_snp, 1))

        hsqhat = reg.Hsq(chisq, ref_ld, s(sumstats[w_ld_cname]), s(sumstats.N),
                         M_annot, n_blocks=n_blocks, intercept=args.intercept_h2,
                         twostep=args.two_step, old_weights=old_weights)

        H2.append(hsqhat)
        _print_h2(args, hsqhat, ref_ld_cnames, k, n_pheno)

        if args.overlap_annot:
            overlap_matrix, M_tot = _read_annot(args)

            df_results = hsqhat._overlap_output(ref_ld_cnames, overlap_matrix, M_annot, M_tot, args.print_coefficients)
            df_results.to_csv(args.out+'.results', sep="\t", index=False)
            logging.info('Results printed to '+args.out+'.results')

    tables = write_h2_table(args, H2, h2_files, sumstats_list)

    if args.print_estimates:
        logging.info('\n Writting h2 estimate tables to text files...')
        tables['h2'].to_csv(args.out + '_h2_estimates.txt', sep='\t', index=False)

    if args.print_cov:
        logging.info('Printing estimation covariance results to files...')
        tables = print_cov(args, tables, H2=H2, h2_files=h2_files)

    if args.print_delete_vals:
        logging.info('Printing jacknife delete values to files...')
        tables = print_delete_vals(args, tables, H2=H2, h2_files=h2_files)

    if args.write_excel:
        logging.info('Writing estimates to excel tables...')
        writer = pd.ExcelWriter(args.out + '_tables.xlsx')
        for tab_name,tab in tables.items():
            tab.to_excel(writer, tab_name)
        writer.save()

    return H2, tables

def estimate_rg(args):

    '''Estimate rg between all pairs of traits.
              
    '''
    #args = copy.deepcopy(args)
    RG = []
    sumstats_list = {}

    if isinstance(args.rg, str): # command line case
        rg_paths, rg_files = _parse_(args.rg) # rows
        rg_list = rg_paths # columns
        n_pheno = len(rg_paths)
    else: # python case
        rg_frames = args.rg # rows
        rg_list = list(rg_frames) # columns
        n_pheno = len(args.rg)
        rg_files =  []
        for i in range(n_pheno):
            rg_files.append("Trait_{}".format(i+1))       

    f = lambda x: _split_or_none(x, n_pheno)
    args.intercept_h2, args.intercept_gencov, args.samp_prev, args.pop_prev = map(f,
        (args.intercept_h2, args.intercept_gencov, args.samp_prev, args.pop_prev))
    map(lambda x: _check_arg_len(x, n_pheno), ((args.intercept_h2, '--intercept-h2'),
                                               (args.intercept_gencov, '--intercept-gencov'),
                                               (args.samp_prev, '--samp-prev'),
                                               (args.pop_prev, '--pop-prev')))
    if args.no_intercept:
        args.intercept_h2 = [1 for _ in range(n_pheno)]
        args.intercept_gencov = [0 for _ in range(n_pheno)]

    ref_ld, w_ld, M_annot, ref_ld_cnames, novar_cols = _read_ref(args)
        
    for k, p1_orig in enumerate(rg_list): # rows

        if isinstance(args.rg, str):
            p1 = p1_orig 
        else:
            p1 = p1_orig.copy()

        sumstats, w_ld_cname = _read_ld_sumstats(args, p1, ref_ld, w_ld, alleles=True, dropna=True)
        n_annot = M_annot.shape[1]
        sumstats_list[k] = pd.DataFrame(sumstats)
        out_prefix = args.out + rg_files[k]

        if n_annot == 1 and args.two_step is None and args.intercept_h2 is None:
            args.two_step = float('inf')
        if args.two_step is not None:
            logging.info('Using two-step estimator with cutoff at {M}.'.format(M=args.two_step))

        for i in range(k, n_pheno): # columns
            p2 = rg_list[i] if type(args.rg) == str else rg_list[i].copy()
            loop = _read_other_sumstats(args, p2, sumstats, ref_ld_cnames, i, k)
            rghat = _rg(loop, args, M_annot, ref_ld_cnames, w_ld_cname, k, i)
            RG.append(rghat)

            if i > k:
                logging.info('Computing rg for phenotypes {K}/{N}-{I}/{N}'.format(K=k+1, I=i+1, N=len(rg_list)))
                _print_gencor(args, rghat, ref_ld_cnames, k,i, rg_list, True)

    X = np.empty(shape=[n_pheno,n_pheno], dtype=object)
    X.fill(np.nan)
    X[np.triu_indices(n_pheno)] = RG
    X[(np.triu_indices(n_pheno)[1],np.triu_indices(n_pheno)[0])] = RG
    RG = X

    tables = write_rg_table(args, RG, n_pheno, rg_files, sumstats_list)

    if args.print_estimates:
        logging.info('\n Writting rg estimate tables to text files...')
        for tab_name,tab in tables.items():
            tab.to_csv(args.out + '_{}'.format(tab_name) + '_estimates.txt', sep='\t', index=False)

    if args.print_cov:
        logging.info('Printing estimation covariance results to files...')
        tables = print_cov(args, tables, RG=RG, rg_files=rg_files)

    if args.print_delete_vals:
        logging.info('Printing jacknife delete values to files...')
        tables = print_delete_vals(args, tables, RG=RG, rg_files=rg_files)        

    if args.write_excel:
        logging.info('Writing estimates to excel tables...')
        writer = pd.ExcelWriter(args.out + '_tables.xlsx')
        for tab_name,tab in tables.items():
            tab.to_excel(writer, tab_name)
        writer.save()

    return RG, tables

def write_h2_table(args, H2, h2_files, sumstats_list):
    logging.info('Formatting heritability estimates ...')
    pd.options.display.float_format = '{:.3f}'.format
    tables = dict()
    tables['h2'] = pd.DataFrame(index=h2_files)

    for i, phen in enumerate(h2_files):
        hsqhat = H2[i]
        tables['h2'].loc[phen, 'SNPs (M)'] = len(sumstats_list[i])   
        tables['h2'].loc[phen, 'h2'] =  hsqhat.tot
        tables['h2'].loc[phen, 'h2_se'] = hsqhat.tot_se
        tables['h2'].loc[phen, 'lambda GC'] = hsqhat.lambda_gc
        tables['h2'].loc[phen, 'Mean chi2'] = hsqhat.mean_chisq
        tables['h2'].loc[phen, 'Intercept'] = hsqhat.intercept
        tables['h2'].loc[phen, 'Intercept_se'] = hsqhat.intercept_se
        if not args.no_intercept and (args.intercept_h2 is None) and (args.intercept_gencov is None):
            tables['h2'].loc[phen, 'Ratio'] = hsqhat.ratio 
            tables['h2'].loc[phen, 'Ratio_SE'] = hsqhat.ratio_se

    return tables

def write_rg_table(args, RG, n_pheno, rg_files, sumstats_list):

    logging.info('Formatting heritability estimates ...')
    tables = dict()
    pd.options.display.float_format = '{:.3f}'.format
    tables['h2'] = pd.DataFrame(index=rg_files)

    for i, phen in enumerate(rg_files):
        ldsc_result = RG[i,i]
        tables['h2'].loc[phen, 'SNPs (M)'] = len(sumstats_list[i])   
        tables['h2'].loc[phen, 'h2'] =  ldsc_result.hsq1.tot
        tables['h2'].loc[phen, 'h2_se'] = ldsc_result.hsq1.tot_se
        tables['h2'].loc[phen, 'lambda GC'] = ldsc_result.hsq1.lambda_gc
        tables['h2'].loc[phen, 'Mean chi2'] = ldsc_result.hsq1.mean_chisq
        tables['h2'].loc[phen, 'Intercept'] = ldsc_result.hsq1.intercept
        tables['h2'].loc[phen, 'Intercept_se'] = ldsc_result.hsq1.intercept_se
        if not args.no_intercept and (args.intercept_h2 is None) and (args.intercept_gencov is None):
            tables['h2'].loc[phen, 'Ratio'] = ldsc_result.hsq1.ratio 
            tables['h2'].loc[phen, 'Ratio_SE'] = ldsc_result.hsq1.ratio_se

    logging.info('Formatting genetic correlation estimates...')
    tables['rg'] = pd.DataFrame(index=rg_files)
    for i, phen1 in enumerate(rg_files): 
        for j, phen2 in enumerate(rg_files):
            if i == j: 
                tables['rg'].loc[phen1,phen2] = '1'
            else:
                tables['rg'].loc[phen1,phen2] = '{:.3f} ({:.3f})'.format(RG[i,j].rg_ratio, RG[i,j].rg_se)

    tables['cov'] = pd.DataFrame(index=rg_files)
    for i, phen1 in enumerate(rg_files): 
        for j, phen2 in enumerate(rg_files):
                tables['cov'].loc[phen1,phen2] = '{:.3f} ({:.3f})'.format(RG[i,j].gencov.tot, RG[i,j].gencov.tot_se)

    return tables

def print_cov(args, tables, H2=None, RG=None, h2_files=None, rg_files=None):
    logging.info('Formatting estimation covariances...')

    if H2 is not None:
        tables['est_cov'] = pd.DataFrame(index=h2_files)
        for i, phen in enumerate(h2_files):
            tables['est_cov'].loc[phen, 'est_cov'] = H2[i].coef_cov.flatten()

    if RG is not None:    
        result_hsq1_cov = matrix_formatter(RG, '.hsq1.coef_cov')
        result_gencov_cov = matrix_formatter(RG, '.gencov.coef_cov')
        tables['est_cov'] = pd.DataFrame(index=rg_files)

        for i, phen1 in enumerate(rg_files): 
            for j, phen2 in enumerate(rg_files):
                if i == j:
                    tables['est_cov'].loc[phen1,phen2] = result_hsq1_cov[i,j]
                else:
                    tables['est_cov'].loc[phen1,phen2] = result_gencov_cov[i,j]

    tables['est_cov'].to_csv(args.out + '_est_cov.txt', sep='\t', index=False)
    return tables

def print_delete_vals(args, tables, H2=None, RG=None, h2_files=None, rg_files=None):
    logging.info('Formatting block jackknife delete-values...')
    if H2 is not None:
        
        tables['est_del'] = pd.DataFrame(columns=h2_files, index=np.arange(args.n_blocks))
        tables['part_del'] = pd.DataFrame(columns=h2_files, index=np.arange(args.n_blocks))

        for i, phen in enumerate(h2_files):
            hsqhat = H2[i]
            est_del = {'Del_Val': hsqhat.tot_delete_values.flatten()}
            part_del = {'Part_Del_Val': hsqhat.part_delete_values.flatten()}
            tables['est_del'][phen] = pd.DataFrame(data=est_del)
            tables['part_del'][phen] = pd.DataFrame(data=part_del)
            _print_part_delete_values(hsqhat, args.out + '.part_delete')

        tables['est_del'].to_csv(args.out + '.del_val', sep='\t', index=False)
        tables['part_del'].to_csv(args.out + '.part_del_val', sep='\t', index=False)

    if RG is not None:
        result_hsq1_del = matrix_formatter(RG, '.hsq1.tot_delete_values')
        result_gencov_del = matrix_formatter(RG, '.gencov.tot_delete_values')
        comb_tags = list(it.combinations_with_replacement(rg_files,2))
        tables['est_del'] = pd.DataFrame(columns=np.array(comb_tags),index=np.arange(args.n_blocks))

        for i, phen1 in enumerate(rg_files): 
            for j, phen2 in enumerate(rg_files):
                if i < j:
                    tables['est_del'].loc[:,comb_tags[int(i*len(rg_files)-i*(i-1)/2+j-i)]] = result_hsq1_del[i,j]
                else:
                    tables['est_del'].loc[:,comb_tags[int(j*len(rg_files)-j*(j-1)/2+i-j)]] = result_gencov_del[i,j]

        tables['est_del'].to_csv(args.out + '.del_val', sep='\t', index=False)

    return tables

def matrix_formatter(result_rg, output_var):
    ''' Key Arguments:
    result_rg - matrix w/ rghat objects obtained from estimate_rg
    output_var - interested variable in the form of '.[VAR_NAME]'
    '''
    output_mat = np.empty_like(result_rg)
    (nrow, ncol) = result_rg.shape
    for i in range(nrow):
        for j in range(ncol):
            if result_rg[i, j] is None:
                output_mat[i, j] = None
            else:
                exec('output_mat[i, j] = result_rg[i, j]{}'.format(output_var))
    return(output_mat)

def _read_other_sumstats(args, fh2, sumstats, ref_ld_cnames, i, k):
    if i > k:
        if type(fh2) == pd.DataFrame:
            loop = fh2.dropna(how='any')
        else:
            loop = _read_sumstats(args, fh2, alleles=True, dropna=True)
        loop = _merge_sumstats_sumstats(args, sumstats, loop)
    else:
        loop = sumstats.copy()
        loop['A1x'] = sumstats.A1
        loop['A2x'] = sumstats.A2
        loop['N2'] = sumstats.N
        loop['Z2'] = sumstats.Z
        loop.rename(columns={'N':'N1','Z':'Z1'}, inplace=True)

    alleles = loop.A1 + loop.A2 + loop.A1x + loop.A2x
    if not args.no_check_alleles:
        loop = _select_and_log(loop, _filter_alleles(alleles),
                               '{N} SNPs with valid alleles.')
    loop['Z2'] = _align_alleles(loop.Z2, alleles)
    loop = loop.drop(['A1', 'A1x', 'A2', 'A2x'], axis=1)
    _check_ld_condnum(args, loop[ref_ld_cnames])
    _warn_length(loop)

    return loop

def _get_rg_table(rg_files, RG, args):
    '''Print a table of genetic correlations.'''
    t = lambda attr: lambda obj: getattr(obj, attr, 'NA')
    x = pd.DataFrame()
    x['p1'] = [rg_files[i][0] for i in range(len(RG))]
    x['p2'] = [rg_path_tups[i][1] for i in range(len(RG))]
    x['rg'] = map(t('rg_ratio'), RG)
    x['se'] = map(t('rg_se'), RG)
    x['z'] = map(t('z'), RG)
    x['p'] = map(t('p'), RG)
    if args.samp_prev is not None and args.pop_prev is not None and\
            all((i is not None for i in args.samp_prev)) and all((i is not None for it in args.pop_prev)):
        c = reg.h2_obs_to_liab(1, args.samp_prev[1], args.pop_prev[1])
        x['h2_liab'] = map(lambda x: c * x, map(t('tot'), map(t('hsq2'), RG)))
        x['h2_liab_se'] = map(
            lambda x: c * x, map(t('tot_se'), map(t('hsq2'), RG)))
    else:
        x['h2_obs'] = map(t('tot'), map(t('hsq2'), RG))
        x['h2_obs_se'] = map(t('tot_se'), map(t('hsq2'), RG))

    x['h2_int'] = map(t('intercept'), map(t('hsq2'), RG))
    x['h2_int_se'] = map(t('intercept_se'), map(t('hsq2'), RG))
    x['gcov_int'] = map(t('intercept'), map(t('gencov'), RG))
    x['gcov_int_se'] = map(t('intercept_se'), map(t('gencov'), RG))
    return x.to_string(header=True, index=False) + '\n'

def _print_gencor(args, rghat, ref_ld_cnames, i1,i2, rg_paths, print_hsq1):
    l = lambda x: x + ''.join(['-' for i in range(len(x.replace('\n', '')))])
    P = [args.samp_prev[i1], args.samp_prev[i2]]
    K = [args.pop_prev[i1], args.pop_prev[i2]]
    if args.samp_prev is None and args.pop_prev is None:
        args.samp_prev = [None, None]
        args.pop_prev = [None, None]
    if print_hsq1:
        logging.info(l('\nHeritability of phenotype {I}/{N}\n'.format(I=i1+1, N=len(rg_paths))))
        logging.info(rghat.hsq1.summary(ref_ld_cnames, P=P[0], K=K[0]))

    logging.info(
        l('\nHeritability of phenotype {I}/{N}\n'.format(I=i2+1, N=len(rg_paths))))
    logging.info(rghat.hsq2.summary(ref_ld_cnames, P=P[1], K=K[1]))
    logging.info(l('\nGenetic Covariance\n'))
    logging.info(rghat.gencov.summary(ref_ld_cnames, P=P, K=K))
    logging.info(l('\nGenetic Correlation\n'))
    logging.info(rghat.summary() + '\n')

def _print_h2(args, hsqhat, ref_ld_cnames, i1, n_pheno):
    l = lambda x: x + ''.join(['-' for i in range(len(x.replace('\n', '')))])
    logging.info(
        l('\nHeritability of phenotype {I}/{N}\n'.format(I=i1+1, N=n_pheno)))
    logging.info(hsqhat.summary(ref_ld_cnames, P=args.samp_prev, K=args.pop_prev, overlap = args.overlap_annot) + '\n')

def _merge_sumstats_sumstats(args, sumstats1, sumstats2):
    '''Merge two sets of summary statistics.'''
    sumstats1.rename(columns={'N': 'N1', 'Z': 'Z1'}, inplace=True)
    sumstats2.rename(
        columns={'A1': 'A1x', 'A2': 'A2x', 'N': 'N2', 'Z': 'Z2'}, inplace=True)
    x = _merge_and_log(sumstats1, sumstats2, 'summary statistics')
    return x


def _filter_alleles(alleles):
    '''Remove bad variants (mismatched alleles, non-SNPs, strand ambiguous).'''
    ii = alleles.apply(lambda y: y in MATCH_ALLELES)
    return ii


def _align_alleles(z, alleles):
    '''Align Z1 and Z2 to same choice of ref allele (allowing for strand flip).'''
    try:
        z *= (-1) ** alleles.apply(lambda y: FLIP_ALLELES[y])
    except KeyError as e:
        msg = 'Incompatible alleles in .sumstats files: %s. ' % e.args
        msg += 'Did you forget to use --merge-alleles with munge_sumstats.py?'
        raise KeyError(msg)
    return z

def _rg(sumstats, args, M_annot, ref_ld_cnames, w_ld_cname, i1, i2):
    '''Run the regressions.'''
    n_snp = len(sumstats)
    s = lambda x: np.array(x).reshape((n_snp, 1))
    if args.chisq_max is not None:
        ii = sumstats.Z1**2*sumstats.Z2**2 < args.chisq_max**2
        n_snp = np.sum(ii)  # lambdas are late binding, so this works
        sumstats = sumstats[ii]
    n_blocks = min(args.n_blocks, n_snp)
    ref_ld = sumstats.as_matrix(columns=ref_ld_cnames)
    intercepts = [args.intercept_h2[i1], args.intercept_h2[i2], args.intercept_gencov[i2]]
    rghat = reg.RG(s(sumstats.Z1), s(sumstats.Z2),
                   ref_ld, s(sumstats[w_ld_cname]), s(
                       sumstats.N1), s(sumstats.N2), M_annot,
                   intercept_hsq1=intercepts[0], intercept_hsq2=intercepts[1],
                   intercept_gencov=intercepts[2], n_blocks=n_blocks, twostep=args.two_step)

    return rghat


def _parse_(filepaths):
    '''Parse args.rg.'''
    paths = _splitp(filepaths)
    files = [x.split('/')[-1] for x in paths]

    return paths, files


def _print_rg_delete_values(rg, fh):
    '''Print block jackknife delete values.'''
    _print_delete_values(rg.hsq1, fh + '.hsq1.delete')
    _print_delete_values(rg.hsq2, fh + '.hsq2.delete')
    _print_delete_values(rg.gencov, fh + '.gencov.delete')


def _print_rg_cov(rghat, fh):
    '''Print covariance matrix of estimates.'''
    _print_cov(rghat.hsq1, fh + '.hsq1.cov')
    _print_cov(rghat.hsq2, fh + '.hsq2.cov')
    _print_cov(rghat.gencov, fh + '.gencov.cov')


def _split_or_none(x, n):
    if x is not None:
        y = map(float, x.replace('N', '-').split(','))
    else:
        y = [None for _ in range(n)]
    return y

def _check_arg_len(x, n):
    x, m = x
    if len(x) != n:
        raise ValueError(
            '{M} must have the same number of arguments as --rg/--h2.'.format(M=m))
