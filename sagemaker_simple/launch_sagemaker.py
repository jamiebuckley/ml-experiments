import os
from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import InputData, SourceCode

data_source = os.environ.get("S3_TRAINING_BUCKET")
sagemaker_role = os.environ.get("SAGEMAKER_ROLE")

train_data = InputData(
    channel_name="training",
    data_source=f"s3://{data_source}"
)

trainer = ModelTrainer(
    training_image="763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/pytorch-training:1.13.1-cpu-py39",
    source_code=SourceCode(
        source_dir=".",
        entry_script="train.py",
        requirements="requirements.txt"
    ),
    hyperparameters={
        "epochs": 10,
        "batchsize": 128,
        "lr": 0.01
    },
    role=sagemaker_role
)

trainer.train(
    input_data_config=[train_data]
)
print(f"Executed training job")
