#!/usr/bin/env bash

# Enlever la ligne suivante si vous n'utilisez pas macOS
export PATH="/opt/homebrew/bin:$PATH" 

URL="http://ai.stanford.edu/~yukez/papers/resources/dataset_v7w_telling.zip"
DOWNLOAD_PATH="dataset_v7w_telling.json"

if [ -f "dataset.json" ]; then
  echo "Dataset already exists."
  read -p "Would you like to update the dataset file? [y/n] " -n 1 -r
  echo    
  if [[ ! $REPLY =~ ^[Yy]$ ]]
  then
    echo "Bye!"
    exit
  fi
fi

echo "Downloading dataset..."
wget -q $URL -O dataset.zip
unzip -j dataset.zip
rm dataset.zip
mv $DOWNLOAD_PATH dataset.json
echo "Done."
