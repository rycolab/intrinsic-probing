#!/bin/bash
EMBEDDING_DIR=../embeddings/fasttext

while IFS=' ' read -r col1 col2
do
  if [ -f $EMBEDDING_DIR/cc.$col2.300.bin.gz ] || [ -f $EMBEDDING_DIR/cc.$col2.300.bin ]; then
    echo "Skipping $col2"
  else
    echo "Downloading $col2..."
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$col2.300.bin.gz -P ../embeddings/fasttext
  fi
done <languages_common_coded.lst
