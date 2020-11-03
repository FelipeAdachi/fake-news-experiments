# fake-news-experiments

Tracking ML experiments with S3 and W&amp;B to train text classification models for fake news detection.

The goal is to have a fully traceable ML Experiment: code, train/test data, models and performance metrics, through different stages: preprocessing, grid search optimization and training final models.

# Uploading Dataset

From the projects root folder:

`python -m src.data.upload_dataset`

This will get external data located at data/external, properly assemble it into a dataframe and split into train/test dataset.

The x/y train-test splits are then uploaded to an S3 bucket, and logged as artifacts at W&B, using references to the bucket. Metadata is included to link bucket versionIDs to the W&B artifacts.

# Grid Search Optimization

`wandb sweep src/models/sweep.yaml`

then run sweep agent given, e.g.

`wandb agent felipeadachi/fn_experiments/<sweep_id>`

This runs a grid search hyperparameter tuning, in order to assess performance for differente sklearn models, such as sdg, svm, random forest and multionomial naive-bayes. An additional parameter related to a text preprocessing step is included, to assess the impact of denoising the text.

# Train Final Model

python -m src.models.train_final

Once the type of model and appropriate text preprocessing is defined, the final model is trained, classification reports and confusion matrix is logged to W&B, joblib model is saved to S3 bucket and referenced to W&B.
