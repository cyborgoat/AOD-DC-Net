#!/bin/sh

train_path="../localfiles/Dataset/devset/medium"
test_path="../localfiles/Dataset/devset/medium"
dev_path="../localfiles/Dataset/devset/medium"
echo "Train Path: $train_path"
echo "Test Path: $test_path"
echo "Dev Path: $dev_path"
python3 data_visualization.py $train_path $test_path $dev_path