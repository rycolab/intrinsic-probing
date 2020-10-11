#!/bin/bash
BATCHID="A"
OPTIONS="--log-wandb"
for LANG_ID in {0..45}; do
  LANGUAGE=$(python scripts/get_ud_treebank_lang_by_id.py $LANG_ID)

  SELECTION_CRITERION="log_likelihood"
  echo "python run_ud_treebanks.py $LANGUAGE bert --use-gpu --tag batch-map-$BATCHID --max-iter 50 --selection-criterion $SELECTION_CRITERION $OPTIONS"
  echo "python run_ud_treebanks.py $LANGUAGE fasttext --use-gpu --tag batch-map-$BATCHID --max-iter 50 --selection-criterion $SELECTION_CRITERION $OPTIONS"
# done | parallel -j 3
done | bash
# done
