import wandb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import boto3
from definitions import ROOT_DIR


tags = ['unprocessed']


wandb.login()
run = wandb.init(project='fn_experiments', job_type='dataset_producer')

### Assembling initial dataframe
data_folder = os.path.join(ROOT_DIR,"data/external")
true = pd.read_csv(os.path.join(data_folder, "True.csv"))
false = pd.read_csv(os.path.join(data_folder, "Fake.csv"))
true['category'] = 1
false['category'] = 0
df = pd.concat([true, false])

df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']

### Splitting into train/test and saving to csv
interim_folder = os.path.join(ROOT_DIR,"data/interim")
x_train_path = os.path.join(interim_folder,'x_train.csv')
x_test_path = os.path.join(interim_folder,'x_test.csv')
y_train_path = os.path.join(interim_folder,'y_train.csv')
y_test_path = os.path.join(interim_folder,'y_test.csv')

x_train, x_test, y_train, y_test = train_test_split(
    df.text, df.category, random_state=0)
x_train.to_csv(x_train_path)
x_test.to_csv(x_test_path)
y_train.to_csv(y_train_path)
y_test.to_csv(y_test_path)

### Uploading to S3 Bucket
s3 = boto3.client('s3')
metadata = {}
bucket = "fn-e2e"
with open(x_train_path, "rb") as f:
    key = "train_test_dataset/x_train.csv"
    response = s3.upload_fileobj(
        f, bucket, key, ExtraArgs={'ACL': 'public-read'})
    obj = s3.get_object(Bucket=bucket, Key=key)
    version_id = obj['VersionId']
    metadata['x_train_version_id'] = version_id
    x_train_url = f'https://{bucket}.s3.amazonaws.com/{key}?versionId={version_id}'
    metadata['x_train_url'] = x_train_url


print("done")
with open(x_test_path, "rb") as f:
    key = "train_test_dataset/x_test.csv"
    response = s3.upload_fileobj(
        f, "fn-e2e", "train_test_dataset/x_test.csv", ExtraArgs={'ACL': 'public-read'})
    obj = s3.get_object(Bucket=bucket, Key=key)
    version_id = obj['VersionId']
    metadata['x_test_version_id'] = version_id
    x_test_url = f'https://{bucket}.s3.amazonaws.com/{key}?versionId={version_id}'
    metadata['x_test_url'] = x_test_url

print("done")
with open(y_train_path, "rb") as f:
    key = "train_test_dataset/y_train.csv"
    response = s3.upload_fileobj(
        f, "fn-e2e", "train_test_dataset/y_train.csv", ExtraArgs={'ACL': 'public-read'})
    obj = s3.get_object(Bucket=bucket, Key=key)
    version_id = obj['VersionId']
    metadata['y_train_version_id'] = version_id
    y_train_url = f'https://{bucket}.s3.amazonaws.com/{key}?versionId={version_id}'
    metadata['y_train_url'] = y_train_url

print("done")
with open(y_test_path, "rb") as f:
    key = "train_test_dataset/y_test.csv"
    response = s3.upload_fileobj(
        f, "fn-e2e", "train_test_dataset/y_test.csv", ExtraArgs={'ACL': 'public-read'})
    obj = s3.get_object(Bucket=bucket, Key=key)
    version_id = obj['VersionId']
    metadata['y_test_version_id'] = version_id
    y_test_url = f'https://{bucket}.s3.amazonaws.com/{key}?versionId={version_id}'
    metadata['y_test_url'] = y_test_url


### Logging dataset artifacts at W&B as S3 object references
artifact = wandb.Artifact(
    'train_test_dataset', type='dataset', metadata=metadata)


artifact.add_reference(
    x_train_url, name='x_train.csv', checksum=True)
artifact.add_reference(
    x_test_url, name='x_test.csv', checksum=True)
artifact.add_reference(
    y_train_url, name='y_train.csv', checksum=True)
artifact.add_reference(
    y_test_url, name='y_test.csv', checksum=True)

run.log_artifact(artifact, aliases=tags)

# end the current run
wandb.join()
