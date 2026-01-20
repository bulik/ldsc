#!/bin/bash

#SBATCH --ntasks=10
#SBATCH --cpus-per-task=30
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_sldsc_h2cts_%j.out
#SBATCH --error=slurm_sldsc_h2cts_%j.err

source /opt/software/anaconda3/2023.03-py3.10/etc/profile.d/conda.sh
conda activate /projects/cbmr_shared/people/wkq953/non-GDPR/.conda/envs/ldsc

# Cell-Type Specific Analysis (LDSC-SEG), Estimate regression coefficient (τ) and P-value.

# === Set project specific paths ===
OUT_DIR="/projects/qtl_families-AUDIT/scratch/sldsc_result/baselineLD_v2.2_h2_cts"
mkdir -p "$OUT_DIR"
ANNOT_DIR="/projects/qtl_families-AUDIT/scratch/ldsc_annotation" # store annotation peak files
LDCTS_FILE="$ANNOT_DIR/celltype.ldcts"

# === Set general paths ===
RESOURCE_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc"
GWAS_DIR="/projects/cbmr_shared/data/common_resources/SEGMENT/GWAS/processed/ldsc_munge"
LDSC_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/ldsc"
baseline_model="$RESOURCE_DIR/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD."
weights_file="$RESOURCE_DIR/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC."
frq_file="$RESOURCE_DIR/1000G_Phase3_frq/1000G.EUR.QC."

cd "$LDSC_DIR" || exit 1

# === Generate cts reference file ===
> "$LDCTS_FILE"
for f in "$ANNOT_DIR"/*.chr1.l2.ldscore.gz; do

    cluster_name=$(basename "$f" .chr1.l2.ldscore.gz)
    prefix_path="${ANNOT_DIR}/${cluster_name}.chr"

    echo -e "$cluster_name\t$prefix_path" >> "$LDCTS_FILE"
done


# === Loop through all GWAS sumstats ===
for trait in "$GWAS_DIR"/*.sumstats.gz; do

    trait_name=$(basename "$trait" .sumstats.gz)
    echo "Processed $trait_name"
    
    python ldsc.py \
        --h2-cts "$trait" \
        --ref-ld-chr-cts "$LDCTS_FILE" \
        --ref-ld-chr "$baseline_model" \
        --w-ld-chr "$weights_file" \
        --frqfile-chr "$frq_file" \
        --out "$OUT_DIR/${trait_name}" 

done


# test
# ./ldsc.py \
#     --h2-cts /projects/cbmr_shared/data/common_resources/SEGMENT/GWAS/processed/ldsc_munge/Alcohol_Linner2019.sumstats.gz \
#     --ref-ld-chr /projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD. \
#     --out test_h2_cts_modify_Alcohol_Linner2019 \
#     --ref-ld-chr-cts /projects/qtl_families-AUDIT/scratch/ldsc_annotation/celltype.ldcts \
#     --frqfile-chr /projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc/1000G_Phase3_frq/1000G.EUR.QC. \
#     --w-ld-chr /projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. 
