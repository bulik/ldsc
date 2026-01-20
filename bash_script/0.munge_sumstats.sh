#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=bash_script/slurm_munge_GWAS_%j.out
#SBATCH --error=bash_script/slurm_munge_GWAS_%j.err

source /opt/software/anaconda3/2023.03-py3.10/etc/profile.d/conda.sh
conda activate /projects/cbmr_shared/people/wkq953/non-GDPR/.conda/envs/ldsc

ALLELE_FILE="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/resources/ldsc/w_hm3.snplist"
OUTPUT_DIR="/projects/cbmr_shared/data/common_resources/SEGMENT/GWAS/processed/ldsc_munge"
INPUT_DIR="/projects/cbmr_shared/data/common_resources/SEGMENT/GWAS/processed/SSF_format"
DEFAULT_DIR="/projects/cbmr_shared/people/wkq953/non-GDPR/segment/pipeline/ldsc"

mkdir -p "$OUTPUT_DIR"
cd "$DEFAULT_DIR"


for file in "$INPUT_DIR"/*.tsv; do

    # Skip finemapped files
    if [[ "$file" == *finemapped* ]]; then
        continue
    fi

    base=$(basename "$file" .tsv)
    outname="$OUTPUT_DIR/${base}"
    sumstats_clean="${OUTPUT_DIR}/${base}.clean.tsv"

    # munge_sumstats will filter out all rows with NA or empty values 
    # Check if it's a binary trait by looking at column 17 (n_cases)
    is_binary=$(awk -F'\t' 'NR>1 && $17 != "NA" && $17 != "" && $17+0 > 0 { print 1; exit } END { print 0 }' "$file"| head -n1)

    echo "Processing: $file"
    echo "Is binary trait: $is_binary"

    if [ "$is_binary" -eq 1 ]; then
        echo "binary trait"
        cut -f3,4,5,7,8,11,14,17,18 "$file" > "$sumstats_clean"
    else
        cut -f3,4,5,7,8,11,14 "$file" > "$sumstats_clean"
    fi

    python munge_sumstats.py \
        --sumstats "$sumstats_clean" \
        --out "$outname" \
        --merge-alleles "$ALLELE_FILE" \
        --chunksize 500000 
    
    rm -f "$sumstats_clean"

done
