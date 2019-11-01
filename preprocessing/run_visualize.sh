#!/bin/sh

train_path="../localfiles/data_demo/"
validation_path="../localfiles/data_demo/"
action="stat" # stat / img
echo "Train Path: $train_path"
echo "Test Path: $validation_path"
python3 data_visualization.py $train_path $validation_path $dev_path $action