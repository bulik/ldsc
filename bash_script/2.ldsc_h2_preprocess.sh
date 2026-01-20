#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=bash_script/slurm_ldsc_h2_preprocess_input_%j.out
#SBATCH --error=bash_script/slurm_ldsc_h2_preprocess_input_%j.err
#SBATCH --array=1-22


# === Set project specific paths ===
PROJECT_DIR="/projects/qtl_families-AUDIT/scratch/ldsc_annotation"
OUTPUT_DIR="/projects/qtl_families-AUDIT/scratch/ldsc_annotation_combine_celltypes"

# === Set general paths ===
CHROM=${SLURM_ARRAY_TASK_ID}  # Auto-index from --array
DEFAULT_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/ldsc"
BASELINE_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc/1000G_Phase3_baselineLD_v2.2_ldscores"

mkdir -p "$OUTPUT_DIR"
cd "$DEFAULT_DIR"

echo "--- Starting LDSC Data Preparation for Chromosome ${CHROM} ---"
echo "Project Directory: ${PROJECT_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Baseline Directory: ${BASELINE_DIR}"


# Step 1 extract only Base from Baseline model

echo "Step 1: Extracting Baseline Model BASE Data for Chr ${CHROM}"

zcat "$BASELINE_DIR/baselineLD.${CHROM}.l2.ldscore.gz" | cut -f1-4 > "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.ldscore"
zcat "$BASELINE_DIR/baselineLD.${CHROM}.annot.gz"      | cut -f1-5 > "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.annot"
cut -f1 "$BASELINE_DIR/baselineLD.${CHROM}.l2.M"       > "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.M"
cut -f1 "$BASELINE_DIR/baselineLD.${CHROM}.l2.M_5_50"  > "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.M_5_50"


# Step 2 combine celltype annotations for each chromosome with baseline model

echo "Step 2: Integrating Cell Type Specific Data for Chr ${CHROM}"

for annot_file in "$PROJECT_DIR"/*.chr${CHROM}.annot.gz; do # e.g. ATAC_peaks_per_cluster_narrowPeaks.real.muscle_9_peaks.chr9.annot.gz

    cluster_full=$(basename "$annot_file" .chr${CHROM}.annot.gz)
    cluster=${cluster_full##*.}  # e.g. muscle_9_peaks

    ldscore_file="$PROJECT_DIR/${cluster_full}.chr${CHROM}.l2.ldscore.gz"
    M_file="$PROJECT_DIR/${cluster_full}.chr${CHROM}.l2.M"
    M_5_50_file="$PROJECT_DIR/${cluster_full}.chr${CHROM}.l2.M_5_50"

    tmp_col=$(mktemp)
    
    # === LD score ===
    {
        echo "${cluster}L2"
        zcat "$ldscore_file" | tail -n +2 | cut -f4
    } > "$tmp_col"
    paste "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.ldscore" "$tmp_col" > "${tmp_col}_new"
    mv "${tmp_col}_new" "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.ldscore"

    # === Annotation ===
    {
        echo "$cluster"
        zcat "$annot_file" | tail -n +2 | cut -f5
    } > "$tmp_col"
    paste "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.annot" "$tmp_col" > "${tmp_col}_new"
    mv "${tmp_col}_new" "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.annot"

    # === M ===
    paste "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.M" "$M_file" > "${tmp_col}_new"
    mv "${tmp_col}_new" "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.M"

    # === M_5_50 ===
    paste "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.M_5_50" "$M_5_50_file" > "${tmp_col}_new"
    mv "${tmp_col}_new" "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.M_5_50"

done

cat "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.ldscore" | gzip -c > "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.ldscore.gz"
cat "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.annot" | gzip -c > "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.annot.gz"
rm "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.l2.ldscore"
rm "$OUTPUT_DIR/baseline_w_celltype_LD.${CHROM}.annot"