#!/usr/bin/env bash

# Enlever la ligne suivante si vous n'utilisez pas macOS
export PATH="/opt/homebrew/bin:$PATH" 

URL="http://vision.stanford.edu/yukezhu/visual7w_images.zip"
DOWNLOAD_PATH="images"

if [ -d "$DOWNLOAD_PATH" ]; then
  echo "Folder 'images' already exists."
  read -p "Would you like to update the Images folder? [y/n] " -n 1 -r
  echo    
  if [[ ! $REPLY =~ ^[Yy]$ ]]
  then
    echo "Donwloading cancelled."
    exit
  fi
  rm -rf "$DOWNLOAD_PATH"
fi

echo "Please note that this download is 1.7GB in size. This may take a while."
echo "Downloading images..."
wget -q $URL -O images.zip
unzip -j images.zip -d $DOWNLOAD_PATH
rm images.zip
echo "Done."