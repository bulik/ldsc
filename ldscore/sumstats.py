'''
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

This module deals with getting all the data needed for LD Score regression from files
into memory and checking that the input makes sense. There is no math here. LD Score
regression is implemented in the regressions module.
'''

import numpy as np
import pandas as pd
from scipy import stats
import itertools as it
from . import parse as ps
from . import regressions as reg
import sys
import traceback
import copy
import os
import glob


_N_CHR = 22
# complementary bases
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
# bases
BASES = list(COMPLEMENT.keys())
# true iff strand ambiguous
STRAND_AMBIGUOUS = {''.join(x): x[0] == COMPLEMENT[x[1]]
                    for x in it.product(BASES, BASES)
                    if x[0] != x[1]}
# SNPS we want to keep (pairs of alleles)
VALID_SNPS = {x for x in [''.join(y) for y in it.product(BASES, BASES)]
              if x[0] != x[1] and not STRAND_AMBIGUOUS[x]}
# T iff SNP 1 has the same alleles as SNP 2 (allowing for strand or ref allele flip).
MATCH_ALLELES = {x for x in [''.join(y) for y in it.product(VALID_SNPS, VALID_SNPS)]
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


def _select_and_log(x, ii, log, msg):
    '''Fiter down to rows that are True in ii. Log # of SNPs removed.'''
    new_len = ii.sum()
    if new_len == 0:
        raise ValueError(msg.format(N=0))
    else:
        x = x[ii]
        log.log(msg.format(N=new_len))
    return x


def smart_merge(x, y):
    '''Check if SNP columns are equal. If so, save time by using concat instead of merge.'''
    if len(x) == len(y) and (x.index == y.index).all() and (x.SNP == y.SNP).all():
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True).drop('SNP', axis=1)
        out = pd.concat([x, y], axis=1)
    else:
        out = pd.merge(x, y, how='inner', on='SNP')
    return out


def _read_ref_ld(args, log):
    '''Read reference LD Scores.'''
    ref_ld = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
                                   'reference panel LD Score', ps.ldscore_fromlist)
    log.log(
        'Read reference panel LD Scores for {N} SNPs.'.format(N=len(ref_ld)))
    return ref_ld


def _read_annot(args, log):
    '''Read annot matrix.'''
    try:
        if args.ref_ld is not None:
            overlap_matrix, M_tot = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
                                                          'annot matrix', ps.annot, frqfile=args.frqfile)
        elif args.ref_ld_chr is not None:
            overlap_matrix, M_tot = _read_chr_split_files(args.ref_ld_chr, args.ref_ld, log,
                                                      'annot matrix', ps.annot, frqfile=args.frqfile_chr)
    except Exception:
        log.log('Error parsing .annot file.')
        raise

    return overlap_matrix, M_tot


def _read_M(args, log, n_annot):
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


def _read_w_ld(args, log):
    '''Read regression SNP LD.'''
    if (args.w_ld and ',' in args.w_ld) or (args.w_ld_chr and ',' in args.w_ld_chr):
        raise ValueError(
            '--w-ld must point to a single fileset (no commas allowed).')
    w_ld = _read_chr_split_files(args.w_ld_chr, args.w_ld, log,
                                 'regression weight LD Score', ps.ldscore_fromlist)
    if len(w_ld.columns) != 2:
        raise ValueError('--w-ld may only have one LD Score column.')
    w_ld.columns = ['SNP', 'LD_weights']  # prevent colname conflicts w/ ref ld
    log.log(
        'Read regression weight LD Scores for {N} SNPs.'.format(N=len(w_ld)))
    return w_ld


def _read_chr_split_files(chr_arg, not_chr_arg, log, noun, parsefunc, **kwargs):
    '''Read files split across 22 chromosomes (annot, ref_ld, w_ld).'''
    try:
        if not_chr_arg:
            log.log('Reading {N} from {F} ... ({p})'.format(N=noun, F=not_chr_arg, p=parsefunc.__name__))
            out = parsefunc(_splitp(not_chr_arg), **kwargs)
        elif chr_arg:
            f = ps.sub_chr(chr_arg, '[1-22]')
            log.log('Reading {N} from {F} ... ({p})'.format(N=noun, F=f, p=parsefunc.__name__))
            out = parsefunc(_splitp(chr_arg), _N_CHR, **kwargs)
    except ValueError as e:
        log.log('Error parsing {N}.'.format(N=noun))
        raise e

    return out


def _read_sumstats(args, log, fh, alleles=False, dropna=False):
    '''Parse summary statistics.'''
    log.log('Reading summary statistics from {S} ...'.format(S=fh))
    sumstats = ps.sumstats(fh, alleles=alleles, dropna=dropna)
    log_msg = 'Read summary statistics for {N} SNPs.'
    log.log(log_msg.format(N=len(sumstats)))
    m = len(sumstats)
    sumstats = sumstats.drop_duplicates(subset='SNP')
    if m > len(sumstats):
        log.log(
            'Dropped {M} SNPs with duplicated rs numbers.'.format(M=m - len(sumstats)))

    return sumstats


def _check_ld_condnum(args, log, ref_ld):
    '''Check condition number of LD Score matrix.'''
    if len(ref_ld.shape) >= 2:
        cond_num = int(np.linalg.cond(ref_ld))
        if cond_num > 100000:
            if args.invert_anyway:
                warn = "WARNING: LD Score matrix condition number is {C}. "
                warn += "Inverting anyway because the --invert-anyway flag is set."
                log.log(warn.format(C=cond_num))
            else:
                warn = "WARNING: LD Score matrix condition number is {C}. "
                warn += "Remove collinear LD Scores. "
                raise ValueError(warn.format(C=cond_num))


def _check_variance(log, M_annot, ref_ld):
    '''Remove zero-variance LD Scores.'''
    ii = ref_ld.iloc[:, 1:].var() == 0  # NB there is a SNP column here
    if ii.all():
        raise ValueError('All LD Scores have zero variance.')
    else:
        log.log('Removing partitioned LD Scores with zero variance.')
        ii_snp = np.array([True] + list(~ii))
        ii_m = np.array(~ii)
        ref_ld = ref_ld.iloc[:, ii_snp]
        M_annot = M_annot[:, ii_m]

    return M_annot, ref_ld, ii


def _warn_length(log, sumstats):
    if len(sumstats) < 200000:
        log.log(
            'WARNING: number of SNPs less than 200k; this is almost always bad.')


def _print_cov(ldscore_reg, ofh, log):
    '''Prints covariance matrix of slopes.'''
    log.log(
        'Printing covariance matrix of the estimates to {F}.'.format(F=ofh))
    np.savetxt(ofh, ldscore_reg.coef_cov)


def _print_delete_values(ldscore_reg, ofh, log):
    '''Prints block jackknife delete-k values'''
    log.log('Printing block jackknife delete values to {F}.'.format(F=ofh))
    np.savetxt(ofh, ldscore_reg.tot_delete_values)

def _print_part_delete_values(ldscore_reg, ofh, log):
    '''Prints partitioned block jackknife delete-k values'''
    log.log('Printing partitioned block jackknife delete values to {F}.'.format(F=ofh))
    np.savetxt(ofh, ldscore_reg.part_delete_values)


def _merge_and_log(ld, sumstats, noun, log):
    '''Wrap smart merge with log messages about # of SNPs.'''
    sumstats = smart_merge(ld, sumstats)
    msg = 'After merging with {F}, {N} SNPs remain.'
    if len(sumstats) == 0:
        raise ValueError(msg.format(N=len(sumstats), F=noun))
    else:
        log.log(msg.format(N=len(sumstats), F=noun))

    return sumstats


def _read_ld_sumstats(args, log, fh, alleles=False, dropna=True):
    sumstats = _read_sumstats(args, log, fh, alleles=alleles, dropna=dropna)
    ref_ld = _read_ref_ld(args, log)
    n_annot = len(ref_ld.columns) - 1
    M_annot = _read_M(args, log, n_annot)
    M_annot, ref_ld, novar_cols = _check_variance(log, M_annot, ref_ld)
    w_ld = _read_w_ld(args, log)
    sumstats = _merge_and_log(ref_ld, sumstats, 'reference panel LD', log)
    sumstats = _merge_and_log(sumstats, w_ld, 'regression SNP LD', log)
    w_ld_cname = sumstats.columns[-1]
    ref_ld_cnames = ref_ld.columns[1:len(ref_ld.columns)]
    return M_annot, w_ld_cname, ref_ld_cnames, sumstats, novar_cols

def cell_type_specific(args, log):
    '''Cell type specific analysis'''
    args = copy.deepcopy(args)
    if args.intercept_h2 is not None:
        args.intercept_h2 = float(args.intercept_h2)
    if args.no_intercept:
        args.intercept_h2 = 1

    M_annot_all_regr, w_ld_cname, ref_ld_cnames_all_regr, sumstats, novar_cols = \
            _read_ld_sumstats(args, log, args.h2_cts)
    M_tot = np.sum(M_annot_all_regr)
    _check_ld_condnum(args, log, ref_ld_cnames_all_regr)
    _warn_length(log, sumstats)
    n_snp = len(sumstats)
    n_blocks = min(n_snp, args.n_blocks)
    if args.chisq_max is None:
        chisq_max = max(0.001*sumstats.N.max(), 80)
    else:
        chisq_max = args.chisq_max

    ii = np.ravel(sumstats.Z**2 < chisq_max)
    sumstats = sumstats.iloc[ii, :]
    log.log('Removed {M} SNPs with chi^2 > {C} ({N} SNPs remain)'.format(
            C=chisq_max, N=np.sum(ii), M=n_snp-np.sum(ii)))
    n_snp = np.sum(ii)  # lambdas are late-binding, so this works
    ref_ld_all_regr = np.array(sumstats[ref_ld_cnames_all_regr]).reshape((len(sumstats),-1))
    chisq = np.array(sumstats.Z**2)
    keep_snps = sumstats[['SNP']]

    s = lambda x: np.array(x).reshape((n_snp, 1))
    results_columns = ['Name', 'Coefficient', 'Coefficient_std_error', 'Coefficient_P_value']
    results_data = []
    for (name, ct_ld_chr) in [x.split() for x in open(args.ref_ld_chr_cts).readlines()]:
        ref_ld_cts_allsnps = _read_chr_split_files(ct_ld_chr, None, log,
                                   'cts reference panel LD Score', ps.ldscore_fromlist)
        log.log('Performing regression.')
        ref_ld_cts = np.array(pd.merge(keep_snps, ref_ld_cts_allsnps, on='SNP', how='left').iloc[:,1:])
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
    log.log('Results printed to '+args.out+'.cell_type_results.txt')


def estimate_h2(args, log):
    '''Estimate h2 and partitioned h2.'''
    args = copy.deepcopy(args)
    if args.samp_prev is not None and args.pop_prev is not None:
        args.samp_prev, args.pop_prev = list(map(
            float, [args.samp_prev, args.pop_prev]))
    if args.intercept_h2 is not None:
        args.intercept_h2 = float(args.intercept_h2)
    if args.no_intercept:
        args.intercept_h2 = 1
    M_annot, w_ld_cname, ref_ld_cnames, sumstats, novar_cols = _read_ld_sumstats(
        args, log, args.h2)
    ref_ld = np.array(sumstats[ref_ld_cnames])
    _check_ld_condnum(args, log, ref_ld_cnames)
    _warn_length(log, sumstats)
    n_snp = len(sumstats)
    n_blocks = min(n_snp, args.n_blocks)
    n_annot = len(ref_ld_cnames)
    chisq_max = args.chisq_max
    old_weights = False
    if n_annot == 1:
        if args.two_step is None and args.intercept_h2 is None:
            args.two_step = 30
    else:
        old_weights = True
        if args.chisq_max is None:
            chisq_max = max(0.001*sumstats.N.max(), 80)

    s = lambda x: np.array(x).reshape((n_snp, 1))
    chisq = s(sumstats.Z**2)
    if chisq_max is not None:
        ii = np.ravel(chisq < chisq_max)
        sumstats = sumstats.iloc[ii, :]
        log.log('Removed {M} SNPs with chi^2 > {C} ({N} SNPs remain)'.format(
                C=chisq_max, N=np.sum(ii), M=n_snp-np.sum(ii)))
        n_snp = np.sum(ii)  # lambdas are late-binding, so this works
        ref_ld = np.array(sumstats[ref_ld_cnames])
        chisq = chisq[ii].reshape((n_snp, 1))

    if args.two_step is not None:
        log.log('Using two-step estimator with cutoff at {M}.'.format(M=args.two_step))

    hsqhat = reg.Hsq(chisq, ref_ld, s(sumstats[w_ld_cname]), s(sumstats.N),
                     M_annot, n_blocks=n_blocks, intercept=args.intercept_h2,
                     twostep=args.two_step, old_weights=old_weights)

    if args.print_cov:
        _print_cov(hsqhat, args.out + '.cov', log)
    if args.print_delete_vals:
        _print_delete_values(hsqhat, args.out + '.delete', log)
        _print_part_delete_values(hsqhat, args.out + '.part_delete', log)

    log.log(hsqhat.summary(ref_ld_cnames, P=args.samp_prev, K=args.pop_prev, overlap = args.overlap_annot))
    if args.overlap_annot:
        overlap_matrix, M_tot = _read_annot(args, log)

        # overlap_matrix = overlap_matrix[np.array(~novar_cols), np.array(~novar_cols)]#np.logical_not
        df_results = hsqhat._overlap_output(ref_ld_cnames, overlap_matrix, M_annot, M_tot, args.print_coefficients)
        df_results.to_csv(args.out+'.results', sep="\t", index=False)
        log.log('Results printed to '+args.out+'.results')

    return hsqhat


def estimate_rg(args, log):
    '''Estimate rg between trait 1 and a list of other traits.'''
    args = copy.deepcopy(args)
    rg_paths, rg_files = _parse_rg(args.rg)
    n_pheno = len(rg_paths)
    f = lambda x: _split_or_none(x, n_pheno)
    args.intercept_h2, args.intercept_gencov, args.samp_prev, args.pop_prev = list(map(f,
        (args.intercept_h2, args.intercept_gencov, args.samp_prev, args.pop_prev)))
    list(map(lambda x: _check_arg_len(x, n_pheno), ((args.intercept_h2, '--intercept-h2'),
                                               (args.intercept_gencov, '--intercept-gencov'),
                                               (args.samp_prev, '--samp-prev'),
                                               (args.pop_prev, '--pop-prev'))))
    if args.no_intercept:
        args.intercept_h2 = [1 for _ in range(n_pheno)]
        args.intercept_gencov = [0 for _ in range(n_pheno)]
    p1 = rg_paths[0]
    out_prefix = args.out + rg_files[0]
    M_annot, w_ld_cname, ref_ld_cnames, sumstats, _ = _read_ld_sumstats(args, log, p1,
                                                                        alleles=True, dropna=True)
    RG = []
    n_annot = M_annot.shape[1]
    if n_annot == 1 and args.two_step is None and args.intercept_h2 is None:
        args.two_step = 30
    if args.two_step is not None:
        log.log('Using two-step estimator with cutoff at {M}.'.format(M=args.two_step))

    for i, p2 in enumerate(rg_paths[1:n_pheno]):
        log.log(
            'Computing rg for phenotype {I}/{N}'.format(I=i + 2, N=len(rg_paths)))
        try:
            loop = _read_other_sumstats(args, log, p2, sumstats, ref_ld_cnames)
            rghat = _rg(loop, args, log, M_annot, ref_ld_cnames, w_ld_cname, i)
            RG.append(rghat)
            _print_gencor(args, log, rghat, ref_ld_cnames, i, rg_paths, i == 0)
            out_prefix_loop = out_prefix + '_' + rg_files[i + 1]
            if args.print_cov:
                _print_rg_cov(rghat, out_prefix_loop, log)
            if args.print_delete_vals:
                _print_rg_delete_values(rghat, out_prefix_loop, log)

        except Exception:  # keep going if phenotype 50/100 causes an error
            msg = 'ERROR computing rg for phenotype {I}/{N}, from file {F}.'
            log.log(msg.format(I=i + 2, N=len(rg_paths), F=rg_paths[i + 1]))
            ex_type, ex, tb = sys.exc_info()
            log.log(traceback.format_exc(ex) + '\n')
            if len(RG) <= i:  # if exception raised before appending to RG
                RG.append(None)

    log.log('\nSummary of Genetic Correlation Results\n' +
            _get_rg_table(rg_paths, RG, args))
    return RG


def _read_other_sumstats(args, log, p2, sumstats, ref_ld_cnames):
    loop = _read_sumstats(args, log, p2, alleles=True, dropna=False)
    loop = _merge_sumstats_sumstats(args, sumstats, loop, log)
    loop = loop.dropna(how='any')
    alleles = loop.A1 + loop.A2 + loop.A1x + loop.A2x
    if not args.no_check_alleles:
        loop = _select_and_log(loop, _filter_alleles(alleles), log,
                               '{N} SNPs with valid alleles.')
        loop['Z2'] = _align_alleles(loop.Z2, alleles)

    loop = loop.drop(['A1', 'A1x', 'A2', 'A2x'], axis=1)
    _check_ld_condnum(args, log, loop[ref_ld_cnames])
    _warn_length(log, loop)
    return loop


def _get_rg_table(rg_paths, RG, args):
    '''Print a table of genetic correlations.'''
    t = lambda attr: lambda obj: getattr(obj, attr, 'NA')
    x = pd.DataFrame()
    x['p1'] = [rg_paths[0] for i in range(1, len(rg_paths))]
    x['p2'] = rg_paths[1:len(rg_paths)]
    x['rg'] = list(map(t('rg_ratio'), RG))
    x['se'] = list(map(t('rg_se'), RG))
    x['z'] = list(map(t('z'), RG))
    x['p'] = list(map(t('p'), RG))
    if args.samp_prev is not None and \
            args.pop_prev is not None and \
            all((i is not None for i in args.samp_prev)) and \
            all((i is not None for it in args.pop_prev)):

        c = list(map(lambda x, y: reg.h2_obs_to_liab(1, x, y), args.samp_prev[1:], args.pop_prev[1:]))
        x['h2_liab'] = list(map(lambda x, y: x * y, c, list(map(t('tot'), list(map(t('hsq2'), RG))))))
        x['h2_liab_se'] = list(map(lambda x, y: x * y, c, list(map(t('tot_se'), list(map(t('hsq2'), RG))))))
    else:
        x['h2_obs'] = list(map(t('tot'), list(map(t('hsq2'), RG))))
        x['h2_obs_se'] = list(map(t('tot_se'), list(map(t('hsq2'), RG))))

    x['h2_int'] = list(map(t('intercept'), list(map(t('hsq2'), RG))))
    x['h2_int_se'] = list(map(t('intercept_se'), list(map(t('hsq2'), RG))))
    x['gcov_int'] = list(map(t('intercept'), list(map(t('gencov'), RG))))
    x['gcov_int_se'] = list(map(t('intercept_se'), list(map(t('gencov'), RG))))
    return x.to_string(header=True, index=False) + '\n'


def _print_gencor(args, log, rghat, ref_ld_cnames, i, rg_paths, print_hsq1):
    l = lambda x: x + ''.join(['-' for i in range(len(x.replace('\n', '')))])
    P = [args.samp_prev[0], args.samp_prev[i + 1]]
    K = [args.pop_prev[0], args.pop_prev[i + 1]]
    if args.samp_prev is None and args.pop_prev is None:
        args.samp_prev = [None, None]
        args.pop_prev = [None, None]
    if print_hsq1:
        log.log(l('\nHeritability of phenotype 1\n'))
        log.log(rghat.hsq1.summary(ref_ld_cnames, P=P[0], K=K[0]))

    log.log(
        l('\nHeritability of phenotype {I}/{N}\n'.format(I=i + 2, N=len(rg_paths))))
    log.log(rghat.hsq2.summary(ref_ld_cnames, P=P[1], K=K[1]))
    log.log(l('\nGenetic Covariance\n'))
    log.log(rghat.gencov.summary(ref_ld_cnames, P=P, K=K))
    log.log(l('\nGenetic Correlation\n'))
    log.log(rghat.summary() + '\n')


def _merge_sumstats_sumstats(args, sumstats1, sumstats2, log):
    '''Merge two sets of summary statistics.'''
    sumstats1.rename(columns={'N': 'N1', 'Z': 'Z1'}, inplace=True)
    sumstats2.rename(
        columns={'A1': 'A1x', 'A2': 'A2x', 'N': 'N2', 'Z': 'Z2'}, inplace=True)
    x = _merge_and_log(sumstats1, sumstats2, 'summary statistics', log)
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


def _rg(sumstats, args, log, M_annot, ref_ld_cnames, w_ld_cname, i):
    '''Run the regressions.'''
    n_snp = len(sumstats)
    s = lambda x: np.array(x).reshape((n_snp, 1))
    if args.chisq_max is not None:
        ii = sumstats.Z1**2*sumstats.Z2**2 < args.chisq_max**2
        n_snp = np.sum(ii)  # lambdas are late binding, so this works
        sumstats = sumstats[ii]
    n_blocks = min(args.n_blocks, n_snp)
    ref_ld = sumstats[ref_ld_cnames].to_numpy()
    intercepts = [args.intercept_h2[0], args.intercept_h2[
        i + 1], args.intercept_gencov[i + 1]]
    rghat = reg.RG(s(sumstats.Z1), s(sumstats.Z2),
                   ref_ld, s(sumstats[w_ld_cname]), s(
                       sumstats.N1), s(sumstats.N2), M_annot,
                   intercept_hsq1=intercepts[0], intercept_hsq2=intercepts[1],
                   intercept_gencov=intercepts[2], n_blocks=n_blocks, twostep=args.two_step)

    return rghat


def _parse_rg(rg):
    '''Parse args.rg.'''
    rg_paths = _splitp(rg)
    rg_files = [x.split('/')[-1] for x in rg_paths]
    if len(rg_paths) < 2:
        raise ValueError(
            'Must specify at least two phenotypes for rg estimation.')

    return rg_paths, rg_files


def _print_rg_delete_values(rg, fh, log):
    '''Print block jackknife delete values.'''
    _print_delete_values(rg.hsq1, fh + '.hsq1.delete', log)
    _print_delete_values(rg.hsq2, fh + '.hsq2.delete', log)
    _print_delete_values(rg.gencov, fh + '.gencov.delete', log)


def _print_rg_cov(rghat, fh, log):
    '''Print covariance matrix of estimates.'''
    _print_cov(rghat.hsq1, fh + '.hsq1.cov', log)
    _print_cov(rghat.hsq2, fh + '.hsq2.cov', log)
    _print_cov(rghat.gencov, fh + '.gencov.cov', log)


def _split_or_none(x, n):
    if x is not None:
        y = list(map(float, x.replace('N', '-').split(',')))
    else:
        y = [None for _ in range(n)]
    return y


def _check_arg_len(x, n):
    x, m = x
    if len(x) != n:
        raise ValueError(
            '{M} must have the same number of arguments as --rg/--h2.'.format(M=m))
