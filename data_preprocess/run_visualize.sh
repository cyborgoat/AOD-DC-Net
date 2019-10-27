#!/bin/sh

train_path="../localfiles/data_demo/"
test_path="../localfiles/data_demo/"
dev_path="../localfiles/data_demo/"
action="img" # stat / img
echo "Train Path: $train_path"
echo "Test Path: $test_path"
echo "Dev Path: $dev_path"
python3 data_visualization.py $train_path $test_path $dev_path $action