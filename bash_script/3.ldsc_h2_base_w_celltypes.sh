#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_sldsc_h2_celltype_%j.out
#SBATCH --error=slurm_sldsc_h2_celltype_%j.err

source /opt/software/anaconda3/2023.03-py3.10/etc/profile.d/conda.sh
conda activate /projects/cbmr_shared/people/wkq953/non-GDPR/.conda/envs/ldsc

# Only Base in Baseline model + all cell-type annotations, captures relative enrichment, adjusting for collinearity.

# === Set project specific paths ===
PROJECT_DIR="/projects/qtl_families-AUDIT/scratch"
CLUSTER=${SLURM_ARRAY_TASK_ID}
ANNOT_DIR="$PROJECT_DIR/ldsc_annotation_base_w_celltypes/baseline_w_celltype_LD."  # ATAC_peaks_per_cluster_narrowPeaks.real.muscle_9_peaks.chr9.l2.M
OUT_DIR="$PROJECT_DIR/sldsc_result/baselineLD_v2.2_h2_base_w_celltypes"
mkdir -p "$OUT_DIR"


# === Set general paths ===
RESOURCE_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc"
GWAS_DIR="/projects/cbmr_shared/data/common_resources/SEGMENT/GWAS/processed/ldsc_munge"
LDSC_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/ldsc"
BASELINE_DIR="$RESOURCE_DIR/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD."
weights_file="$RESOURCE_DIR/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC."
frq_file="$RESOURCE_DIR/1000G_Phase3_frq/1000G.EUR.QC."

cd "$LDSC_DIR" || exit 1

# === Loop through all GWAS sumstats ===
for trait in "$GWAS_DIR"/*.sumstats.gz; do

    trait_name=$(basename "$trait" .sumstats.gz)
    echo "Processed $trait_name"
    
    python ldsc.py \
        --h2 "$trait" \
        --ref-ld-chr "$ANNOT_DIR" \
        --w-ld-chr "$weights_file" \
        --frqfile-chr "$frq_file" \
        --overlap-annot \
        --out "$OUT_DIR/${trait_name}" 
        # --overlap-annot --print-cov --print-coefficients --print-delete-vals \   

done
