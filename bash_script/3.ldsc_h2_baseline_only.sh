#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_sldsc_h2_baselinemodel_%j.out
#SBATCH --error=slurm_sldsc_h2_baselinemodel_%j.err

source /opt/software/anaconda3/2023.03-py3.10/etc/profile.d/conda.sh
conda activate /projects/cbmr_shared/people/wkq953/non-GDPR/.conda/envs/ldsc

# this script is for calculatet the h2 based on full baseline model across all traits then filter out traits with z-score<7 as the Finucane,2015 publicatin sugguested to ensure sufficient polygenic signal.


# === Set general paths ===
GWAS_DIR="/projects/cbmr_shared/data/common_resources/SEGMENT/GWAS/processed/ldsc_munge"
OUT_DIR="$GWAS_DIR/h2_baseline"
mkdir -p "$OUT_DIR"
LDSC_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/ldsc"
RESOURCE_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc"
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
        --ref-ld-chr "$BASELINE_DIR" \
        --w-ld-chr "$weights_file" \
        --frqfile-chr "$frq_file" \
        --overlap-annot \
        --out "$OUT_DIR/${trait_name}" 
        # --overlap-annot --print-cov --print-coefficients --print-delete-vals \   

done

OUTPUT=${OUT_DIR}/all_traits_ldsc_qc_summary.tsv
echo -e "Trait\th2\tSE\tZscore\tRatio\tRatio_SE\tIntercept\tIntercept_SE\tLambda_GC\tMean_Chi2" > "$OUTPUT"

for log_file in "$OUT_DIR"/*.log; do

    trait_name=$(basename "$log_file" .log)
    echo "Extract h2 information from $trait_name"
    
    lambda_gc=$(grep "Lambda GC" "$log_file" | awk '{print $3}')
    mean_chi2=$(grep "Mean Chi" "$log_file" | awk '{print $3}')
    
    intercept_line=$(grep "Intercept" "$log_file")
    intercept=$(echo "$intercept_line" | awk '{print $2}')
    intercept_se=$(echo "$intercept_line" | awk '{print $3}' | tr -d '()')

    ratio_line=$(grep "Ratio" "$log_file")
    ratio=$(echo "$ratio_line" | awk '{print $2}')
    ratio_se=$(echo "$ratio_line" | awk '{print $3}' | tr -d '()')

    h2_line=$(grep "Total Observed scale h2:" "$log_file")
    h2=$(echo "$h2_line" | awk '{print $5}')
    h2_se=$(echo "$h2_line" | awk '{print $6}' | tr -d '()')
    zscore=$(awk -v a="$h2" -v b="$h2_se" 'BEGIN {if (b>0) printf "%.2f", a/b; else print "NA"}')

    echo -e "${trait_name}\t${h2}\t${h2_se}\t${zscore}\t${ratio}\t${ratio_se}\t${intercept}\t${intercept_se}\t${lambda_gc}\t${mean_chi2}" >> "$OUTPUT"

done