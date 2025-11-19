import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


def load_data():      
    df = pd.read_csv('marketing_campaign.csv', sep=';')

    # Make column names and values look uniform
    df.columns = df.columns.str.lower()

    categorical_cols = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_cols:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    # convert year_birth column to age
    df['age'] = 2024 - df['year_birth']

    # then drop year_birth
    del df['year_birth']

    # convert dt_customer to customer tenure 
    df['customer_days'] = (pd.to_datetime('today') - pd.to_datetime(df['dt_customer'])).dt.days

    # drop dt_customer
    del df['dt_customer']

    # checking for redundant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            df.drop(col, axis=1, inplace=True)

    df.marital_status.unique()

    df = df[~df['marital_status'].isin(['alone', 'absurd', 'yolo'])]

    # creating aggregated columns for extra features and better marketing info:

    # create total purchase
    purchase_cols = ['numdealspurchases', 'numwebpurchases', 'numcatalogpurchases',
                    'numstorepurchases']

    df['total_purchases'] = df[purchase_cols].sum(axis=1)

    # create total spending
    spending_cols = ['mntwines', 'mntfruits', 'mntmeatproducts',
                    'mntfishproducts', 'mntsweetproducts', 'mntgoldprods']

    df['total_spending'] = df[spending_cols].sum(axis=1)


    # create previous campaign response rate
    df['previous_response_rate'] = df[['acceptedcmp1', 'acceptedcmp2', 'acceptedcmp3', 
                                        'acceptedcmp4', 'acceptedcmp5']].sum(axis=1) / 5
    df['income'] = df['income'].fillna(0)

    return df




# ### Setting up the validation framework
def split_data(df, test_size=0.2, random_state=1):

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


    len(df_train), len(df_val), len(df_test)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.response.values
    y_test = df_test.response.values
    y_val = df_val.response.values

    del df_train['response']
    del df_test['response']
    del df_val['response']

    return df_train, df_val, df_test, y_train, y_val, y_test



def train_model(df_train, y_train):
    """Train XGBoost model with pipeline"""
    xgb_model = XGBClassifier(
        eta=0.1,
        max_depth=10,
        min_child_weight=1,
        subsample=0.5,
        colsample_bytree=1.0,
        reg_lambda=0,
        reg_alpha=0,
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=200,
        nthread=8,
        random_state=1,
        seed=1,
        verbosity=0
    )

    
    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        xgb_model
    )
    
    train_dict = df_train.to_dict(orient='records')
    pipeline.fit(train_dict, y_train)
    
    return pipeline

def evaluate_model(pipeline, df_test, y_test):
    """Evaluate model on test set"""
    test_dict = df_test.to_dict(orient='records')
    y_pred = pipeline.predict_proba(test_dict)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred)
    
    return test_auc


def save_model(pipeline, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    
    print(f'Model is saved to {filename}')




df = load_data()

df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)

pipeline = train_model(df_train, y_train)

auc = evaluate_model(pipeline, df_test, y_test)
print("Test AUC:", auc)

save_model(pipeline, "model.bin")