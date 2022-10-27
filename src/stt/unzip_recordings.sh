#!/bin/bash

dest_folder=data/raw/stt/recordings/

echo $dest_folder

rm -r $dest_folder

mkdir $dest_folder

unzip data/raw/stt/all_recordings.zip -d $dest_folder

echo "File unzipped. All files loaded"
