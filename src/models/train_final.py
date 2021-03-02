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
import pickle
from wandb import AlertLevel
from datetime import timedelta


def train_final_model(data, model_type):
    config = wandb.config
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    x_train = x_train.apply(denoise.denoise_text)
    x_test = x_test.apply(denoise.denoise_text)

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts = count_vect.fit_transform(x_train)
    vocab = count_vect.vocabulary_
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
    f1_score = metrics.f1_score(y_test,preds)
    precision = metrics.precision_score(y_test,preds)
    recall = metrics.recall_score(y_test,preds)
    wandb.log({'f1-score':f1_score,'precision':precision,'recall':recall}) 
    if 'threshold' in config:
        if f1_score < config.threshold:
            wandb.alert(
            title='Low F1 Score',
            text=f'F1 Score {f1_score} is below the acceptable theshold of {config.threshold}',
            level=AlertLevel.WARN,
            wait_duration=timedelta(minutes=0)
            )
    wandb.sklearn.plot_confusion_matrix(y_test, preds)

    return clf,vocab

def main():
    model_type = 'svm'
    model_path = os.path.join(ROOT_DIR,'models')

    tags = [model_type]

    run = wandb.init(project='fn_experiments', job_type='final_model_trainer')

    artifact = run.use_artifact('felipeadachi/fn_experiments/train_test_dataset:v0', type='dataset')
    # # artifact_dir = artifact.download()

    metadata = artifact.metadata

    x_train_csv = requests.get(metadata['x_train_url']).content
    y_train_csv = requests.get(metadata['y_train_url']).content
    x_test_csv = requests.get(metadata['x_test_url']).content
    y_test_csv = requests.get(metadata['y_test_url']).content

    x_train = pd.read_csv(io.BytesIO(x_train_csv),encoding='utf-8',engine='python')['text']
    y_train = pd.read_csv(io.BytesIO(y_train_csv),encoding='utf-8',engine='python')['category']
    x_test = pd.read_csv(io.BytesIO(x_test_csv),encoding='utf-8',engine='python')['text']
    y_test = pd.read_csv(io.BytesIO(y_test_csv),encoding='utf-8',engine='python')['category']
    # print("reading csvs.....")
    # x_train = pd.read_csv('x_train.csv')['text']
    # y_train = pd.read_csv('y_train.csv')['category']
    # x_test = pd.read_csv('x_test.csv')['text']
    # y_test = pd.read_csv('y_test.csv')['category']


    data = {}
    # data['x_train'] = x_train
    # data['y_train'] = y_train
    # data['x_test'] = x_test
    # data['y_test'] = y_test

    data['x_train'] = x_train.sample(n=200)
    data['y_train'] = y_train.sample(n=200)
    data['x_test'] = x_test.sample(n=200)
    data['y_test'] = y_test.sample(n=200)

    x_train.sample(n=200).to_csv("final_x_train.csv")
    y_train.sample(n=200).to_csv("final_y_train.csv")
    x_test.sample(n=200).to_csv("final_x_test.csv")
    y_test.sample(n=200).to_csv("final_y_test.csv")
    run_name = wandb.run.name
    filename = '{}.joblib'.format(run_name)
    feature_name = 'feature_{}.pickle'.format(run_name)
    print("training final models.....")
    clf,vocab = train_final_model(data, model_type)
    pickle.dump(vocab,open(os.path.join(model_path,"feature_"+run_name+".pickle"),"wb"))


    ### Saving model to local
    model_path = os.path.join(ROOT_DIR,'models')
    joblib.dump(clf, os.path.join(model_path, filename))

    ### Uploading model to S3 Bucket
    s3 = boto3.client('s3')
    metadata = {}
    feature_metadata = {}
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

    with open(os.path.join(model_path, feature_name), "rb") as f:
        key = "models/{}".format(feature_name)

        response = s3.upload_fileobj(
            f, bucket, key, ExtraArgs={'ACL': 'public-read'})
        obj = s3.get_object(Bucket=bucket, Key=key)
        version_id = obj['VersionId']
        metadata['feature_version_id'] = version_id
        feature_url = f'https://{bucket}.s3.amazonaws.com/{key}?versionId={version_id}'
        metadata['feature_url'] = feature_url

    ### Logging model artifact in W&B with reference to S3 Bucket
    artifact = wandb.Artifact(
        'model', type='model', metadata=metadata)

    artifact.add_reference(
        model_url, name=filename, checksum=True)

    artifact.add_reference(
        feature_url, name=feature_name, checksum=True)

    run.log_artifact(artifact, aliases=tags)
    # end the current run
    wandb.join()


# nbClassifier(data, denoise)
if __name__ == "__main__":
    main()