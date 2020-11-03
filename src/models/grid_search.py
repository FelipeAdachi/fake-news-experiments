import argparse
import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import wandb
from sklearn import metrics
from sklearn.model_selection import KFold
from ..features import denoise
import requests
import io


def train_and_log(data, model, denoise="False"):
    X = data['x_train']
    y = data['y_train']
    # x_test = data['x_test']
    # y_test = data['y_test']
    num_splits = 5
    kf = KFold(n_splits=num_splits)
    kf.get_n_splits(X)
    global_acc = 0
    for train_index, test_index in kf.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if denoise == "True":
            x_train = x_train.apply(denoise.denoise_text)
            x_test = x_test.apply(denoise.denoise_text)

        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        X_train_counts = count_vect.fit_transform(x_train)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        if model == 'nb':
            from sklearn.naive_bayes import MultinomialNB
            clf = MultinomialNB().fit(X_train_tfidf, y_train)
        if model == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=10).fit(X_train_tfidf, y_train)
        if model == 'sgd':
            from sklearn.linear_model import SGDClassifier
            clf = SGDClassifier(loss='hinge', penalty='l2',
                                alpha=1e-3, random_state=42,
                                max_iter=5, tol=None).fit(X_train_tfidf, y_train)
        if model == 'svc':
            from sklearn import svm
            clf = svm.SVC(kernel='linear').fit(X_train_tfidf, y_train)

        X_test_counts = count_vect.transform(x_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)

        preds = clf.predict(X_test_tfidf)
        # pred_prob = clf.predict_proba(X_test_tfidf)
        print(metrics.classification_report(y_test, preds))
        global_acc = global_acc + float(metrics.accuracy_score(y_test, preds))

    global_acc = global_acc/num_splits
    wandb.log({'accuracy_score': global_acc})

    # wandb.sklearn.plot_classifier(clf, X_train_tfidf, X_test_idf, y_train, y_test, preds,
    #                               pred_prob, clf.classes_, model_name='MultinomialNB', feature_names=None)


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='foo help')
parser.add_argument('--denoise', help='foo help')

args = parser.parse_args()

model = args.model
denoise = args.denoise

print("Model>>>", model)
print("Denoise>>", denoise)

run = wandb.init(project='fn_experiments', job_type='grid_search')

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
# data['x_train'] = x_train
# data['y_train'] = y_train
# data['x_test'] = x_test
# data['y_test'] = y_test

data['x_train'] = x_train.sample(n=200)
data['y_train'] = y_train.sample(n=200)
data['x_test'] = x_test.sample(n=200)
data['y_test'] = y_test.sample(n=200)
print(data['x_train'].head())

train_and_log(data, model, denoise)
