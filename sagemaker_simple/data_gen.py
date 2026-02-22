# Generates some test/train data and uploads to S3
import os
import numpy as np
import boto3
import json

from config import ROOT_DIR
import pandas as pd

data_source = os.environ.get("S3_TRAINING_BUCKET")

s3 = boto3.client('s3')
pd.set_option('display.max_columns', None)

raw_data_path = os.path.join(ROOT_DIR, '.rawdata', 'acorn_sat', 'sagemaker')
acorn_parquet_path = os.path.join(ROOT_DIR, '.rawdata', 'acorn_sat', 'max.parquet')

df = pd.read_parquet(acorn_parquet_path)
df = df[['days_since_start', 'max']]
df = df.iloc[::100]

col_stats = {}
for col in df.columns.tolist():
    col_stats[col + "_min"] = c_min = df[col].min()
    col_stats[col + "_min"] = c_max = df[col].max()
    df[col] = (df[col] - c_min) / (c_max - c_min)

x_values = df['days_since_start'].values.astype('float32')
y_values = df['max'].values.astype('float32')

indices = np.random.permutation(len(x_values))
split = int(0.8 * len(x_values))

train_x, test_x = x_values[indices[:split]], x_values[indices[split:]]
train_y, test_y = y_values[indices[:split]], y_values[indices[split:]]

print(f"X train data: {len(train_x)}, length of test data: {len(test_x)}")
print(f"Y train data: {len(train_y)}, length of test data: {len(test_y)}")

if not os.path.exists(raw_data_path):
    os.makedirs(raw_data_path)

if not os.path.exists(os.path.join(raw_data_path, 'train')):
    os.makedirs(os.path.join(raw_data_path, 'train'))

if not os.path.exists(os.path.join(raw_data_path, 'test')):
    os.makedirs(os.path.join(raw_data_path, 'test'))

np.save(os.path.join(raw_data_path, 'train', 'x_train.npy'), train_x)
np.save(os.path.join(raw_data_path, 'train', 'y_train.npy'), train_y)
np.save(os.path.join(raw_data_path, 'test', 'x_test.npy'), test_x)
np.save(os.path.join(raw_data_path, 'test', 'y_test.npy'), test_y)
with open(os.path.join(raw_data_path, 'stats.json'), 'w') as f:
    json.dump(col_stats, f, indent=4, default=str)


s3.upload_file(os.path.join(raw_data_path, 'stats.json'), data_source, 'stats.json')
s3.upload_file(os.path.join(raw_data_path, 'train', 'x_train.npy'), data_source, 'train/x_train.npy')
s3.upload_file(os.path.join(raw_data_path, 'train', 'y_train.npy'), data_source, 'train/y_train.npy')
s3.upload_file(os.path.join(raw_data_path, 'test', 'x_test.npy'), data_source, 'test/x_test.npy')
s3.upload_file(os.path.join(raw_data_path, 'test', 'y_test.npy'), data_source, 'test/y_test.npy')
print("Data uploaded successfully.")