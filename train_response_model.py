#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[88]:


df = pd.read_csv('marketing_campaign.csv', sep=';')
df.head()


# In[89]:


#check dataset size
df.shape


# ### Data Preparation

# In[90]:


# Make column names and values look uniform
df.columns = df.columns.str.lower()

categorical_cols = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_cols:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# In[91]:


# convert year_birth column to age
df['age'] = 2024 - df['year_birth']

# then drop year_birth
del df['year_birth']


# In[92]:


# convert dt_customer to customer tenure 
df['customer_days'] = (pd.to_datetime('today') - pd.to_datetime(df['dt_customer'])).dt.days

# drop dt_customer
del df['dt_customer']


# In[93]:


df.columns


# In[94]:


# checking for redundant columns
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"DROP: {col} - only has 1 unique value: {df[col].unique()[0]}")


# In[95]:


df.drop(['z_costcontact', 'z_revenue'], axis=1, inplace=True)


# In[96]:


df.marital_status.unique()


# In[97]:


df = df[~df['marital_status'].isin(['alone', 'absurd', 'yolo'])]


# Creating aggregated columns for extra features and better marketing info:

# In[98]:


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


# In[99]:


df.head()


# In[100]:


df.complain.unique()


# In[101]:


df.isnull().sum()


# In[102]:


df['income'] = df['income'].fillna(0)


# ### Setting up the validation framework

# In[103]:


from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[104]:


len(df_train), len(df_val), len(df_test)


# In[105]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[106]:


y_train = df_train.response.values
y_test = df_test.response.values
y_val = df_val.response.values


# In[107]:


del df_train['response']
del df_test['response']
del df_val['response']


# ### EDA

# In[108]:


df_full_train = df_full_train.reset_index(drop=True)
df_full_train.response.value_counts(normalize=True)


# Rate of response to last marketing campaign:

# In[109]:


global_response_rate = df_full_train.response.mean()
round(global_response_rate, 2)


# In[110]:


df.dtypes


# In[111]:


df.columns


# In[112]:


categorical = ['education', 'marital_status']

numerical = ['income', 'kidhome', 'teenhome',
       'recency', 'mntwines', 'mntfruits', 'mntmeatproducts',
       'mntfishproducts', 'mntsweetproducts', 'mntgoldprods',
       'numdealspurchases', 'numwebpurchases', 'numcatalogpurchases',
       'numstorepurchases', 'numwebvisitsmonth', 'acceptedcmp3',
       'acceptedcmp4', 'acceptedcmp5', 'acceptedcmp1', 'acceptedcmp2',
       'complain', 'age', 'customer_days', 'total_purchases',
       'total_spending', 'previous_response_rate']


# ### Feature Importance

# **Risk Ratio**

# In[113]:


from IPython.display import display


# In[114]:


for c in categorical:
    df_group = df_full_train.groupby(c).response.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_response_rate
    df_group['ratio'] = df_group['mean'] / global_response_rate
    display(df_group)
    print()
    print()


# **Mutual Information**

# In[115]:


from sklearn.metrics import mutual_info_score


# In[116]:


def mutual_info_response_score(series):
    return mutual_info_score(series, df_full_train.response)


# In[117]:


mi = df_full_train[categorical].apply(mutual_info_response_score)
mi.sort_values(ascending=False)


# **Correlation**

# In[118]:


correlation = df_full_train[numerical].corrwith(df_full_train.response)
correlation.sort_values(ascending=False)


# Customers who responded to past campaigns are much more likely to respond again (previous_response_rate -0.432156)
# Recent campaign acceptance is highly predictive acceptedcmp5 (0.339) - Second best
# total_spending (0.272) - High spenders are more responsive
# 
# Risk Ratio Insights
# Education:
# 
# PhD holders: 1.64x more likely to respond (best segment!) while Basic education: 0.39x (much less likely). The higher the education, the better the chances of responding.
# 
# Marital Status:
# Single: 1.24x more likely

# ### One Hot Encoding

# In[119]:


from sklearn.feature_extraction import DictVectorizer


# In[120]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient = 'records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


# ### Logistic Regression

# In[121]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)


# In[122]:


y_pred = model.predict_proba(X_val)[:, 1]
y_pred[:10]


# In[123]:


# check model for overfitting
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)

print(f"Training Accuracy: {train_score:.4f}")
print(f"Val Accuracy: {val_score:.4f}")
print(f"Difference: {train_score - val_score:.4f}")


# In[124]:


# checking model acccuracy at threshold 0.5

response_decision = (y_pred >= 0.5)
df_pred = pd.DataFrame({
    'probability': y_pred,
    'prediction': response_decision.astype(int),
    'actual_value': y_val
})
df_pred['correct'] = (df_pred['prediction'] == df_pred['actual_value'])

df_pred.head(10)  


# In[125]:


print(len(y_val))

# ratio of accurate predictions
print((y_val == response_decision).mean())

# number of accurate predictions at t=0.5
print((y_val == response_decision).sum())


# ### Hyperparameters tuning

# **Checking model accuracy at other thresholds:**

# In[126]:


from sklearn.metrics import accuracy_score

thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    response_decision = (y_pred >= t)
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' %(t, score))
    scores.append(score)


# In[127]:


#checking for data imbalance:
y_val.mean()


# Data is imbalanced. only 15% of customers respond, the remaining 85% do not. In this case, the accuracy score doesn't tell us how accurate the model is. Logistic regression will be biased toward the majority class.

# ### Calculating precision, recall and F1 score at different thresholds

# In[128]:


from sklearn.metrics import precision_score, recall_score, f1_score


# In[129]:


precision_list = []
recall_list = []
f1_list = []

thresholds = np.linspace(0, 1, 21)

for t in thresholds:
    y_pred_t = (y_pred >= t).astype(int)
    
    precision_list.append(precision_score(y_val, y_pred_t, zero_division=0))
    recall_list.append(recall_score(y_val, y_pred_t, zero_division=0))
    f1_list.append(f1_score(y_val, y_pred_t, zero_division=0))


# In[130]:


df_scores = pd.DataFrame({
    'thresholds': thresholds,
    'precision': precision_list,
    'recall': recall_list,
    'f1_score': f1_list
})

df_scores


# In[131]:


plt.figure(figsize=(8, 5))
plt.plot(df_scores['thresholds'], df_scores['f1_score'], marker='o', label='F1 Score', linewidth=2)

# Highlight best point
best_idx = df_scores['f1_score'].idxmax()
best_threshold = df_scores.loc[best_idx, 'thresholds']
best_f1 = df_scores.loc[best_idx, 'f1_score']

plt.scatter(best_threshold, best_f1, color='red', s=100, zorder=5, label=f'Best Threshold = {best_threshold:.2f}\nF1 = {best_f1:.3f}')

plt.title('F1 Score vs Decision Threshold', fontsize=14)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# In[132]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(df_scores['thresholds'], df_scores['precision'], marker='o', label='Precision')
plt.plot(df_scores['thresholds'], df_scores['recall'], marker='o', label='Recall')
plt.plot(df_scores['thresholds'], df_scores['f1_score'], marker='o', label='F1 Score', linewidth=2)

# Highlight best F1 point
best_idx = df_scores['f1_score'].idxmax()
best_threshold = df_scores.loc[best_idx, 'thresholds']
best_f1 = df_scores.loc[best_idx, 'f1_score']

plt.scatter(best_threshold, best_f1, color='red', s=100, zorder=5,
            label=f'Best F1 = {best_f1:.3f} at t={best_threshold:.2f}')

plt.title('Precision, Recall and F1 Score Across Thresholds', fontsize=14)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.show()


# ### Confusion Matrix

# In[133]:


from sklearn.metrics import confusion_matrix

# choose your threshold
t = 0.20

# convert probabilities → binary predictions
y_pred_t = (y_pred >= t).astype(int)

# confusion matrix
cm = confusion_matrix(y_val, y_pred_t)

cm


# In[134]:


cm_df = pd.DataFrame(
    cm,
    index=['Actual 0 (No Response)', 'Actual 1 (Response)'],
    columns=['Predicted 0', 'Predicted 1']
)

cm_df


# In[135]:


(cm / cm.sum()).round(2)


# Our model:
# 
# Is good at avoiding wasted spend (high TN, low FP)
# 
# Has moderate ability to find responders (TP is decent but FN is also sizeable)
# 
# Has a reasonable precision (because FP is low)
# 
# Has moderate recall (because FN isn’t tiny)
# 
# If the business cares more about:
# 
# Minimizing wasted marketing cost → this model is good
# 
# Capturing every possible responder → we need a lower threshold to increase recall

# ### ROC Curve and ROC AUC

# In[136]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[137]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)

auc = roc_auc_score(y_val, y_pred)
print(f'Auc Score:{auc}')

# Plot ROC

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], color='gray')  

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ### Hyperparameter tuning with k-fold cross validation

# In[138]:


# function for model training:
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X_train, y_train)

    return dv, model


# In[139]:


# prediction function:
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[140]:


from sklearn.model_selection import KFold


# In[141]:


n_splits = 5
C_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]

for C in C_values:
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_fold_train = df_full_train.iloc[train_idx]
        df_fold_val = df_full_train.iloc[val_idx]

        y_fold_train = df_fold_train.response.values
        y_fold_val = df_fold_val.response.values

        dv, model = train(df_fold_train, y_fold_train, C=C)
        y_fold_pred = predict(df_fold_val, dv, model)

        auc = roc_auc_score(y_fold_val, y_fold_pred)
        scores.append(auc)

    print(f'C={C} {np.mean(scores):.3f} +- {np.std(scores):.3f}')


# In[142]:


#final model training:
dv, model = train(df_full_train, df_full_train.response.values, C=1.0)
y_pred = predict(df_val, dv, model)

auc = roc_auc_score(y_val, y_pred)
auc


# ### Random Forest Model

# In[143]:


from sklearn.ensemble import RandomForestClassifier


# In[144]:


rf = RandomForestClassifier(n_estimators=10, random_state=1)
rf.fit(X_train, y_train)


# In[145]:


y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)


# **Tuning the random forest model:**

# Evaluating the performance of the model as the number of estimators change:

# In[146]:


scores = []

for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    scores.append((n, auc))


# In[147]:


df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])
df_scores


# In[148]:


plt.plot(df_scores.n_estimators, df_scores.auc)
plt.show()


# Evaluating the performance of the model as number of estimators, max_depth change:

# In[149]:


scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
        rf.fit(X_train, y_train)
    
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
    
        scores.append((d, n, auc))


# In[150]:


df_scores = pd.DataFrame(scores, columns=['max_depth', 'n_estimators', 'auc'])
df_scores.head()


# In[151]:


for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth==d]
    plt.plot(df_subset.n_estimators, df_subset.auc, 
             label = f'max_depth={d}')

plt.legend()
plt.show()


# Evaluating different min_leaf_samples values:

# In[152]:


max_depth = 15


# In[153]:


scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, min_samples_leaf=s, random_state=1)
        rf.fit(X_train, y_train)
    
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
    
        scores.append((s, n, auc))


# In[154]:


columns=['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.head()


# In[155]:


for s in [1, 3, 5, 10, 50]:
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    plt.plot(df_subset.n_estimators, df_subset.auc, 
             label = f'min_samples_leaf = {s}')

plt.legend()
plt.show()


# In[156]:


min_samples_leaf = 3

#final model:

rf = RandomForestClassifier(n_estimators=n, 
                            max_depth=max_depth, 
                            min_samples_leaf=min_samples_leaf, 
                            random_state=1)
rf.fit(X_train, y_train)


# In[157]:


y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)


# ### XGBoost Model

# In[158]:


get_ipython().system('pip install xgboost')


# In[159]:


import xgboost as xgb


# In[160]:


# create DMatrix
features = dv.get_feature_names_out().tolist()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


# In[161]:


watchlist=[(dtrain, 'train'), (dval, 'val')]


# In[176]:


xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1, 
    'objective': 'binary:logistic',           
    'nthread': 8, 
    'seed': 1, 
    'verbosity':1,
    'eval_metric': 'auc'
    
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)


# In[177]:


y_pred = model.predict(dval)


# In[178]:


roc_auc_score(y_val, y_pred)


# Evaluate model on 200 trees, print out results in a dataframe, and visualize results:

# In[182]:


evals_result = {}
model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=200,
    evals=watchlist,
    evals_result=evals_result,
    verbose_eval=False,
)

df_score = pd.DataFrame({
    'num_iter': list(range(200)),
    'train_auc': evals_result['train']['auc'],
    'val_auc': evals_result['val']['auc']
})
df_score


# In[181]:


# plt.plot(df_score.num_iter, df_score.train_auc, label='train')
plt.plot(df_score.num_iter, df_score.train_auc, label='train')
plt.plot(df_score.num_iter, df_score.val_auc, label='val')
plt.legend()
plt.show()


# ### **XGBoost parameters tuning:**
# - **Tuning eta:**

# In[ ]:


etas = [0.1, 0.05, 0.01]
scores = {}

for eta in etas:
    xgb_params = {
        'eta': eta,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 0
    }
    
    evals_result = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    # Store results with eta as key
    key = f"eta={eta}"
    scores[key] = pd.DataFrame({
        'num_iter': list(range(200)),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc']
    })
    
    print(f"{key}: Best AUC = {max(evals_result['val']['auc']):.4f}")


# In[ ]:


# Plot different etas
for eta_key in scores.keys():
    df_score = scores[eta_key]
    plt.plot(df_score.num_iter, df_score.val_auc, label=eta_key)
plt.xlabel('Iteration')
plt.ylabel('Validation AUC')
plt.legend()
plt.show()


# eta = 0.1

# **Tuning max_depth:**

# In[ ]:


max_depth = [3, 5, 8, 10]
scores = {}

for m in max_depth:
    xgb_params = {
        'eta': 0.1,
        'max_depth': m,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 0
    }
    
    evals_result = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    # Store results with eta as key
    key = f"max_depth={m}"
    scores[key] = pd.DataFrame({
        'num_iter': list(range(200)),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc']
    })
    
    print(f"{key}: Best AUC = {max(evals_result['val']['auc']):.4f}")


# In[ ]:


for max_depth_key in scores.keys():
    df_score = scores[max_depth_key]
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth_key)
plt.xlabel('Iteration')
plt.ylabel('Validation AUC')
plt.legend()
plt.show()


# **Max_depth = 10**

# **Tuning min_child_weight**

# In[ ]:


min_child_weight = [1, 30, 10]
scores = {}

for m in min_child_weight:
    xgb_params = {
        'eta': 0.1,
        'max_depth': 10,
        'min_child_weight': m,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 0
    }
    
    evals_result = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    # Store results with eta as key
    key = f"min_child_weight={m}"
    scores[key] = pd.DataFrame({
        'num_iter': list(range(200)),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc']
    })
    
    print(f"{key}: Best AUC = {max(evals_result['val']['auc']):.4f}")


# **min_child_weight = 1**

# In[ ]:


for min_child_weight in scores.keys():
    df_score = scores[min_child_weight]
    plt.plot(df_score.num_iter, df_score.val_auc, label=min_child_weight)
plt.xlabel('Iteration')
plt.ylabel('Validation AUC')
plt.legend()
plt.show()


# **Tuning subsample_values:**

# In[ ]:


subsample_values = [0.5, 0.7, 0.9, 1.0]
scores = {}

for s in subsample_values:
    xgb_params = {
        'eta': 0.1,
        'max_depth': 10,
        'min_child_weight': 1,
        'subsample': s,
        'colsample_bytree': 1.0,
        'lambda': 1,
        'alpha': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 0
    }

    evals_result = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )

    key = f"subsample={s}"
    scores[key] = pd.DataFrame({
        'num_iter': range(200),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc'],
    })

    print(f"{key}: Best val AUC = {max(evals_result['val']['auc']):.4f}")


# **Subsample = 0.5**

# **Tuning colsample_values:**

# In[ ]:


colsample_values = [0.3, 0.5, 0.7, 1.0]
scores = {}

for c in colsample_values:
    xgb_params = {
        'eta': 0.1,
        'max_depth': 10,
        'min_child_weight': 1,
        'subsample': 0.5,
        'colsample_bytree': c,
        'lambda': 1,
        'alpha': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 0
    }

    evals_result = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )

    key = f"colsample_bytree={c}"
    scores[key] = pd.DataFrame({
        'num_iter': range(200),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc'],
    })

    print(f"{key}: Best val AUC = {max(evals_result['val']['auc']):.4f}")


# **Tuning lambda_values:**

# In[ ]:


lambda_values = [0, 1, 5, 10]
scores = {}

for l in lambda_values:
    xgb_params = {
        'eta': 0.1,
        'max_depth': 10,
        'min_child_weight': 1,
        'subsample': 0.5,
        'colsample_bytree': 1.0,
        'lambda': l,
        'alpha': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 0
    }

    evals_result = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )

    key = f"lambda={l}"
    scores[key] = pd.DataFrame({
        'num_iter': range(200),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc'],
    })

    print(f"{key}: Best val AUC = {max(evals_result['val']['auc']):.4f}")


# In[ ]:


alpha_values = [0, 0.5, 1, 2, 5]
scores = {}

for a in alpha_values:
    xgb_params = {
        'eta': 0.1,
        'max_depth': 10,
        'min_child_weight': 1,
        'subsample': 0.5,
        'colsample_bytree': 1.0,
        'lambda': 0,
        'alpha': a,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 0
    }

    evals_result = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )

    key = f"alpha={a}"
    scores[key] = pd.DataFrame({
        'num_iter': range(200),
        'train_auc': evals_result['train']['auc'],
        'val_auc': evals_result['val']['auc'],
    })

    print(f"{key}: Best val AUC = {max(evals_result['val']['auc']):.4f}")


# **Final XGBoost Model:**

# In[ ]:


xgb_params = {
    'eta': 0.1,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 1.0,
    'lambda': 0,
    'alpha': 0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 0
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200)

y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)


# ### SELECTING THE BEST MODEL; LogisticRegression, Random forest, XGBoost

# **Best LogisticRegression Model:**

# In[ ]:


dv, model = train(df_full_train, df_full_train.response.values, C=1.0)
y_pred = predict(df_val, dv, model)

auc = roc_auc_score(y_val, y_pred)
auc


# **Best Random forest model:**

# In[ ]:


rf = RandomForestClassifier(n_estimators=200, 
                            max_depth=15, 
                            min_samples_leaf= 3, 
                            random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)


# **Best XGBoost model:**

# In[ ]:


xgb_params = {
    'eta': 0.1,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 1.0,
    'lambda': 0,
    'alpha': 0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 0
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200)

y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)


# **Best Model: XGBoost**

# Training best model on full training dataset:

# In[ ]:


df_full_train


# In[ ]:


y_full_train = df_full_train.response.values
del df_full_train['response']


# In[ ]:


dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)


# In[ ]:


feature_names = dv.get_feature_names_out().tolist()

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, feature_names=feature_names)


# In[ ]:


xgb_params = {
    'eta': 0.1,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 1.0,
    'lambda': 0,
    'alpha': 0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 0
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=200)


# In[ ]:


y_pred = model.predict(dtest)
roc_auc_score(y_test, y_pred)


# Create model pipeline:

# In[ ]:


from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier


# In[ ]:


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
    nthread=8,
    random_state=1,
    verbosity=0
)

pipeline_xgb = make_pipeline(
    DictVectorizer(),
    xgb_model
)


# In[ ]:


train_dict = df_train.to_dict(orient='records')


# In[ ]:


pipeline_xgb.fit(train_dict, y_train)


# In[ ]:


# try out pipeline
sample = df_val.iloc[10].to_dict()
sample


# In[ ]:


#edit the sample above to create a customer
customer = {
    'id': 202,
     'education': 'masters',
     'marital_status': 'widow',
     'income': 82032.0,
     'kidhome': 0,
     'teenhome': 0,
     'recency': 54,
     'mntwines': 332,
     'mntfruits': 194,
     'mntmeatproducts': 377,
     'mntfishproducts': 149,
     'mntsweetproducts': 125,
     'mntgoldprods': 57,
     'numdealspurchases': 0,
     'numwebpurchases': 4,
     'numcatalogpurchases': 6,
     'numstorepurchases': 7,
     'numwebvisitsmonth': 1,
     'acceptedcmp3': 0,
     'acceptedcmp4': 0,
     'acceptedcmp5': 0,
     'acceptedcmp1': 0,
     'acceptedcmp2': 0,
     'complain': 0,
     'age': 76,
     'customer_days': 4245,
     'total_purchases': 17,
     'total_spending': 1234,
     'previous_response_rate': 0.0
}


# In[ ]:


y_pred = pipeline_xgb.predict_proba([customer])[0, 1]
y_pred


# Save model to pickle:

# In[ ]:


import pickle


# In[ ]:


with open ('model.bin', 'wb') as f_out:
    pickle.dump(pipeline_xgb,  f_out)

with open ('model.bin', 'rb') as f_in:
    pipeline_xgb = pickle.load(f_in)


# In[ ]:




