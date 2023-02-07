#!/usr/bin/env python

import pandas as pd
import numpy as np
import argparse
from pybedtools import BedTool
import gzip

def gene_set_to_bed(args):
    print('making gene set bed file')
    GeneSet = pd.read_csv(args.gene_set_file, header = None, names = ['GENE'])
    all_genes = pd.read_csv(args.gene_coord_file, delim_whitespace = True)
    df = pd.merge(GeneSet, all_genes, on = 'GENE', how = 'inner')
    df['START'] = np.maximum(1, df['START'] - args.windowsize)
    df['END'] = df['END'] + args.windowsize
    iter_df = [['chr'+(str(x1).lstrip('chr')), x2 - 1, x3] for (x1,x2,x3) in np.array(df[['CHR', 'START', 'END']])]
    return BedTool(iter_df).sort().merge()

def make_annot_files(args, bed_for_annot):
    print('making annot file')
    df_bim = pd.read_csv(args.bimfile,
            delim_whitespace=True, usecols = [0,1,2,3], names = ['CHR','SNP','CM','BP'])
    iter_bim = [['chr'+str(x1), x2 - 1, x2] for (x1, x2) in np.array(df_bim[['CHR', 'BP']])]
    bimbed = BedTool(iter_bim)
    annotbed = bimbed.intersect(bed_for_annot)
    bp = [x.start + 1 for x in annotbed]
    df_int = pd.DataFrame({'BP': bp, 'ANNOT':1})
    df_annot = pd.merge(df_bim, df_int, how='left', on='BP')
    df_annot.fillna(0, inplace=True)
    df_annot = df_annot[['ANNOT']].astype(int)
    if args.annot_file.endswith('.gz'):
        with gzip.open(args.annot_file, 'wb') as f:
            df_annot.to_csv(f, sep = "\t", index = False)
    else:
        df_annot.to_csv(args.annot_file, sep="\t", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene-set-file', type=str, help='a file of gene names, one line per gene.')
    parser.add_argument('--gene-coord-file', type=str, default='ENSG_coord.txt', help='a file with columns GENE, CHR, START, and END, where START and END are base pair coordinates of TSS and TES. This file can contain more genes than are in the gene set. We provide ENSG_coord.txt as a default.')
    parser.add_argument('--windowsize', type=int, help='how many base pairs to add around the transcribed region to make the annotation?')
    parser.add_argument('--bed-file', type=str, help='the UCSC bed file with the regions that make up your annotation')
    parser.add_argument('--nomerge', action='store_true', default=False, help='don\'t merge the bed file; make an annot file wi    th values proportional to the number of intervals in the bedfile overlapping the SNP.')
    parser.add_argument('--bimfile', type=str, help='plink bim file for the dataset you will use to compute LD scores.')
    parser.add_argument('--annot-file', type=str, help='the name of the annot file to output.')

    args = parser.parse_args()

    if args.gene_set_file is not None:
        bed_for_annot = gene_set_to_bed(args)
    else:
        bed_for_annot = BedTool(args.bed_file).sort()
        if not args.nomerge:
            bed_for_annot = bed_for_annot.merge()

    make_annot_files(args, bed_for_annot)
