# Standard
import os
import json
import pickle

# Third-party
from google.cloud import bigquery
import pandas as pd
import great_expectations as ge
import wandb
from sklearn.model_selection import train_test_split
import joblib

# Local application
from definitions import TEMP_DIR
from . import lakefs_connector
from ..models import train_final



def script_path(filename):
    """
    Function to find the file in the current directory even if it is called from another directory.
    """
    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


def get_data_from_production(bq_client):
    QUERY = (

        'SELECT *'
        'FROM `sunny-emissary-293912.fakenewsdeploy.model_predictions`'
        'WHERE ground_truth in ("Real","Fake") ')
    df = bq_client.query(QUERY).to_dataframe()
    return df

def assert_data_quality(df):
    assert df.expect_column_values_to_be_in_set("ground_truth",["Fake","Real"])["success"]
    assert df.expect_column_values_to_be_unique("url")["success"]
    assert df.expect_column_values_to_not_be_null("content")["success"]
    assert df.expect_column_values_to_be_between("coverage",min_value=0.6,max_value=1)['success']
    assert df.expect_column_values_to_be_between("word_count",min_value=100)['success']

    return df

def upload_prediction_data(lake_conn,run_name,df,filename):
    predictions_path = script_path("online_predictions.csv")
    destination_path = "data/external/online_predictions.csv" 
    df.to_csv(predictions_path)
    res = lake_conn.upload_file(run_name,predictions_path,destination_path)
    return res

def save_expectations_to_wandb(df):
    with open( "my_expectation_file.json", "w") as my_file:
        my_file.write(
            json.dumps(df.get_expectation_suite().to_json_dict())
        )
    wandb.save('my_expectation_file.json')

def merge_online_predictions(true_df,fake_df,df):
    for index,row in df.iterrows():
        if row['ground_truth'] == "Fake":
            new_row = {'text':row['content']}
            fake_df = fake_df.append(new_row,ignore_index=True)
        else:
            new_row = {'text':row['content']}
            true_df = true_df.append(new_row,ignore_index=True)
    return true_df,fake_df

def get_train_test(fake,true):
    fake['category'] = 0
    true['category'] = 1
    df = pd.concat([true, fake])
    df['text'] = df['text'] + " " + df['title']

    del df['title']
    del df['subject']
    del df['date']

    x_train, x_test, y_train, y_test = train_test_split(
        df.text, df.category, random_state=42)
    to_return = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return to_return

def upload_train_test(lake_conn,run_name,train_test):
    for key in train_test:
        keyfile_path = os.path.join(script_path(TEMP_DIR),'{}.csv'.format(key))
        train_test[key].to_csv(keyfile_path)
        destination_path = "data/interim/{}.csv".format(key)
        lake_conn.upload_file(run_name,keyfile_path,destination_path)

def upload_model_files(lake_conn,run_name,vocab,clf):
    feature_name = 'feature_{}.pickle'.format(run_name)
    feature_path = os.path.join(script_path(TEMP_DIR),feature_name)
    destination_path = "models/{}/{}".format(run_name,feature_name)
    pickle.dump(vocab,open(feature_path,"wb"))

    lake_conn.upload_file(run_name,feature_path,destination_path)


    model_name = '{}.joblib'.format(run_name)
    model_path = os.path.join(script_path(TEMP_DIR), model_name)
    destination_path = "models/{}/{}".format(run_name,model_name)
    joblib.dump(clf, model_path)

    lake_conn.upload_file(run_name,model_path,destination_path)

def main():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = script_path('creds.json')
    bq_client = bigquery.Client()
    model_type = 'svm'

    # Create lakefs connector
    lake_conn = lakefs_connector.lakefs_conn()

    # Init wandb run
    wandb.init(project='fn_experiments', job_type='retrain',config={"threshold":0.9})

    run_name = wandb.run.name
    df = get_data_from_production(bq_client)
    df = df.drop_duplicates(subset=['url'])
    df = df[df['word_count']>=100]
    df = ge.from_pandas(df)


    assert_data_quality(df)

    lake_conn.create_branch_from_master(run_name)

    upload_prediction_data(lake_conn,run_name,df,"online_predictions.csv")

    save_expectations_to_wandb(df)


    fake_path = "data/external/Fake.csv"
    fake_df = lake_conn.get_csv(fake_path)

    true_path = "data/external/True.csv"
    true_df = lake_conn.get_csv(true_path)


    true_df,fake_df = merge_online_predictions(true_df,fake_df,df)
    train_test = get_train_test(fake=fake_df.sample(n=40),true=true_df.sample(n=40))
    

    upload_train_test(lake_conn,run_name,train_test)   

    clf,vocab = train_final.train_final_model(train_test,model_type)

    upload_model_files(lake_conn,run_name,vocab,clf)

    commit_message = "Added online_predictions, model files and train test splits for branch {}".format(run_name)
    lake_conn.commit_to_branch(run_name,commit_message)


if __name__ == "__main__":
    main()