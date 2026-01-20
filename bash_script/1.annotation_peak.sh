#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=bash_script/slurm_annotation_%j.out
#SBATCH --error=bash_script/slurm_annotation_%j.err
#SBATCH --array=1-22

# === Load environment ===
source /opt/software/anaconda3/2023.03-py3.10/etc/profile.d/conda.sh
conda activate /projects/cbmr_shared/people/wkq953/non-GDPR/.conda/envs/ldsc

# === Set project specific paths ===
PEAK_DIR="/projects/qtl_families-AUDIT/data/multiome/processed/pipeline_output_freezes/muscle/2.11.0/objects/real/"
ANNOT_DIR="/projects/qtl_families-AUDIT/scratch/ldsc_annotation" # store annotation peak files
mkdir -p "$ANNOT_DIR"

# === Set general paths ===
CHROM=${SLURM_ARRAY_TASK_ID}  # Auto-index from --array
PLINK_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc/1000G_EUR_Phase3_plink"
#PLINK_PREFIX="$PLINK_DIR/1000G.EUR.QC.CHR_1_22"
HM3_SNPLIST="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc/w_hm3.snplist"
LDSC_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/ldsc"

cd "$LDSC_DIR" || exit 1

# === Process each BED file ===

#for PEAK_FILE in "$PEAK_DIR"/*.BED; do
for PEAK_FILE in "$PEAK_DIR"/*.narrowPeak; do

    BASE=$(basename "$PEAK_FILE" .narrowPeak)
    ANNOT_FILE="$ANNOT_DIR/${BASE}.chr${CHROM}.annot.gz"
    OUT_PREFIX="$ANNOT_DIR/${BASE}.chr${CHROM}"

    echo "[chr$CHROM] Generating annotation for $BASE"
    python make_annot.py \
        --bed-file "$PEAK_FILE" \
        --bimfile "$PLINK_DIR/1000G.EUR.QC.${CHROM}.bim" \
        --annot-file "$ANNOT_FILE"

    echo "[chr$CHROM] Computing LD score for $BASE"

    python ldsc.py \
        --l2 \
        --ld-wind-cm 1 \
        --bfile "$PLINK_DIR/1000G.EUR.QC.${CHROM}" \
        --annot "$ANNOT_FILE" \
        --thin-annot \
        --out "$OUT_PREFIX" \
        --print-snps "$HM3_SNPLIST" 
        
done
