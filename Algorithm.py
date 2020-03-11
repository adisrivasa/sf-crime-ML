import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize, scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import pickle

def main():
    train = pd.read_csv("train_final.csv")
    test = pd.read_csv("test_final.csv")

    cat_encoder = LabelEncoder()
    train["CategoryEncoded"] = cat_encoder.fit_transform(train["Category"])

    train_columns = list(train.columns[2:21].values)
    print(train_columns)
    test_columns = list(test.columns[1:].values)
    print(test_columns)

    training, validation = train_test_split(train, train_size=0.7)

    xgb_model = XGBClassifier(n_estimators=100,
                          learning_rate=0.30,
                          max_depth=10,
                          min_child_weight=4,
                          gamma=0.4,
                          reg_alpha=0.05,
                          reg_lambda=2,
                          subsample=0.8,
                          colsample_bytree=1.0,
                          max_delta_step=1,
                          scale_pos_weight=1,
                          objective='multi:softprob',
                          # eval_metric = 'mlogloss',
                          n_jobs = -1,
                          seed=4,
                          silent = 0
                          )
    print("Training...")
    xgb_model.fit(train[train_columns], train["Category"])

    filename = 'final_model.sav'
    pickle.dump(xgb_model, open(filename, 'wb'))

    print("Predicting...")
    predict = xgb_model.predict_proba(test[train_columns])

    print("Validating...")
    validate = xgb_model.predict_proba(validation[train_columns])

    print("Calculating Log Loss")
    logloss = log_loss(validation["Category"], validate)
    print(logloss)

    res = pd.DataFrame(predict, columns=cat_encoder.classes_)
    result = pd.concat([test['Id'], res], axis=1)
    print(result.head())
    result.to_csv("Submission.csv", index=False)

if __name__ == '__main__':
    main()
