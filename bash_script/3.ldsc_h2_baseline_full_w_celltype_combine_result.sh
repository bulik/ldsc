#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_sldsc_h2_full_w_celltype_combine%j.out
#SBATCH --error=slurm_sldsc_h2_full_w_celltype_combine%j.err


# Baseline + one cell-type annotation at a time, captures marginal enrichment, more sensitive but less specific.
# combine all cell-type annotations in one run, which is more efficient than running each cell-type separately.


# === Set project specific paths ===
PROJECT_DIR="/projects/qtl_families-AUDIT/scratch"
GWAS_DIR="/projects/cbmr_shared/data/common_resources/SEGMENT/GWAS/processed/ldsc_munge"
OUT_DIR="$PROJECT_DIR/sldsc_result/baselineLD_v2.2_h2_baseline_full_w_celltypes"


# === Loop through all GWAS sumstats ===
for sumstat_file in "$GWAS_DIR"/*.sumstats.gz; do
    
    trait_name=$(basename "$sumstat_file" .sumstats.gz)
    output_file="${OUT_DIR}/${trait_name}.results"
    echo -e "Category\tProp._SNPs\tProp._h2\tProp._h2_std_error\tEnrichment\tEnrichment_SE\tEnrichment_p\tCoefficient\tCoefficient_std_error\tCoefficient_z-score" > "$output_file"

    for file in ${OUT_DIR}/${trait_name}_cluster*.results; do
        
        [[ -f "$file" ]] || continue
        cluster=$(basename "$file" .results | sed -E 's/.*_cluster([0-9]+)/cluster\1/')
        result_line=$(awk 'NR==2' "$file")
        echo -e "${cluster}\t${result_line#*L2_0}" >> "$output_file"

    done
done
