# Basic Model Training on Sagemaker

Sagemaker's documentation is frankly a bit terrible.

The V3 SDK involves significant changes from V2, so most examples online
don't work.

This is an effort to create a template that works well for training models locally,
i.e. with smaller datasets to verify everything works, then deploy to Sagemaker.


Running on sagemaker requires the following environment variables to be set:
```
S3_TRAINING_BUCKET - Bucket for train/test data
SAGEMAKER_ROLE - Sagemaker execution role for training, s3 access etc.
```