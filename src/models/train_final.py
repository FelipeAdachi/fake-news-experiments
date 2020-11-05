import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import wandb
from sklearn import metrics
import joblib
from sklearn.naive_bayes import MultinomialNB
import boto3
from ..features import denoise
import requests
import io
from definitions import ROOT_DIR


def train_final_model(data, model_type):
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    x_train = x_train.apply(denoise.denoise_text)
    x_test = x_test.apply(denoise.denoise_text)

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts = count_vect.fit_transform(x_train)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    if model_type == 'svm':
        clf = svm.SVC(kernel='linear', probability=True).fit(
            X_train_tfidf, y_train)
    elif model_type == 'nb':
        clf = MultinomialNB().fit(X_train_tfidf, y_train)

    X_test_counts = count_vect.transform(x_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    preds = clf.predict(X_test_tfidf)
    print(metrics.classification_report(y_test, preds))

    wandb.sklearn.plot_confusion_matrix(y_test, preds)

    return clf


model_type = 'svm'
tags = [model_type]

run = wandb.init(project='fn_experiments', job_type='final_model_trainer')

artifact = run.use_artifact('felipeadachi/fn_experiments/train_test_dataset:v0', type='dataset')
# artifact_dir = artifact.download()

metadata = artifact.metadata

x_train_csv = requests.get(metadata['x_train_url']).content
y_train_csv = requests.get(metadata['y_train_url']).content
x_test_csv = requests.get(metadata['x_test_url']).content
y_test_csv = requests.get(metadata['y_test_url']).content

x_train = pd.read_csv(io.BytesIO(x_train_csv),encoding='utf-8')['text']
y_train = pd.read_csv(io.BytesIO(y_train_csv),encoding='utf-8')['category']
x_test = pd.read_csv(io.BytesIO(x_test_csv),encoding='utf-8')['text']
y_test = pd.read_csv(io.BytesIO(y_test_csv),encoding='utf-8')['category']

data = {}
data['x_train'] = x_train
data['y_train'] = y_train
data['x_test'] = x_test
data['y_test'] = y_test

# data['x_train'] = x_train.sample(n=200)
# data['y_train'] = y_train.sample(n=200)
# data['x_test'] = x_test.sample(n=200)
# data['y_test'] = y_test.sample(n=200)

run_name = wandb.run.name
filename = '{}.joblib'.format(run_name)
clf = train_final_model(data, model_type)

### Saving model to local
model_path = os.path.join(ROOT_DIR,'models')
joblib.dump(clf, os.path.join(model_path, filename))

### Uploading model to S3 Bucket
s3 = boto3.client('s3')
metadata = {}

bucket = "fn-e2e"
with open(os.path.join(model_path, filename), "rb") as f:
    key = "models/{}".format(filename)

    response = s3.upload_fileobj(
        f, bucket, key, ExtraArgs={'ACL': 'public-read'})
    obj = s3.get_object(Bucket=bucket, Key=key)
    version_id = obj['VersionId']
    metadata['model_version_id'] = version_id
    model_url = f'https://{bucket}.s3.amazonaws.com/{key}?versionId={version_id}'
    metadata['model_url'] = model_url

### Logging model artifact in W&B with reference to S3 Bucket
artifact = wandb.Artifact(
    'model', type='model', metadata=metadata)

artifact.add_reference(
    model_url, name=filename, checksum=True)
run.log_artifact(artifact, aliases=tags)
# end the current run
wandb.join()


# nbClassifier(data, denoise)
