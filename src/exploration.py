import pandas as pd
import numpy as np
import sys
import os
import random

from itertools import combinations
from statsmodels import robust
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from xgboost import plot_tree
import shap

from itertools import cycle
from scipy import interp

file_features= open("features-2.txt","w+")
file_auc= open("auc-2.txt","w+")

# Parameters
LABEL_COLUMN_NAME = 'satisfaction'
UNWANTED_COLUMNS = ['host_response_rate', 'host_listings_count',
       'host_total_listings_count', 'host_has_profile_pic',
       'host_identity_verified', 'is_location_exact', 'accommodates',
       'bathrooms', 'bedrooms', 'beds',
       'cleaning_fee', 'guests_included', 'minimum_nights',
       'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 
       'has_availability', 'availability_60', 
       'availability_90', 'availability_365',
       'requires_license', 'is_business_travel_ready',
       'require_guest_profile_picture', 'require_guest_phone_verification',
       'calculated_host_listings_count_shared_rooms',
       'host_verifications_count',
       'host_response_time_a few days or more',
       'host_response_time_host_response_time',
       'host_response_time_within a day',
       'host_response_time_within a few hours',
       'host_response_time_within an hour',
       'cancellation_policy_cancellation_policy',
       'cancellation_policy_flexible', 'cancellation_policy_moderate',
       'cancellation_policy_strict',
       'cancellation_policy_strict_14_with_grace_period',
       'cancellation_policy_super_strict_30',
       'cancellation_policy_super_strict_60', 'satisfaction',
       'cancellation_policy_flexible_new', 'cancellation_policy_moderate_new',
       'cancellation_policy_strict_new',
       'cancellation_policy_super_strict_30_new',
       'cancellation_policy_super_strict_60_new']

WANTED_COLUMNS = [
    'review_scores_accuracy',
    'review_scores_value',
    'review_scores_cleanliness',
    'review_scores_communication',
    'host_is_superhost',
    'review_scores_checkin',
    'number_of_reviews',
    'calculated_host_listings_count',
    'price',
    'review_scores_location',
    'number_of_reviews_ltm',
    'reviews_per_month',
    'amenities_count',
    'calculated_host_listings_count_entire_homes',
    'availability_30',
    'maximum_nights',
    'security_deposit',
    'instant_bookable',
    'extra_people',
    'calculated_host_listings_count_private_rooms']

N_FOLDS = 5
RANDOM_STATE = 1

n_estimators = 20
subsample = 0.60
lr = 0.1
max_depth = 10

total = 0
best_models = 0
best_generated_model = 0
feat = []

for c in range(1,50):
    feat.append('feature')

def random_combinations(iterable, r, x):
    pool = tuple(iterable)
    n = len(pool)
    a = []
    for i in range(x):
        indices = sorted(random.sample(range(n), r))
        a.insert(len(a),tuple(pool[i] for i in indices))
    return list(set(a))

def eval_features(df, features):
    global total
    global best_models

    total = total + 1

    X = df[features].values
    y = df[LABEL_COLUMN_NAME].values
    a = []
    b = []
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for (train, val) in cv.split(X, y):
        classifier = XGBClassifier(n_estimators=n_estimators, subsample=subsample, learning_rate=lr, max_depth=max_depth, n_jobs=16, random_state=1)

        classifier = classifier.fit(X[train], y[train])

        probas_ = classifier.predict_proba(X[val])
        area = roc_auc_score(y[val], probas_[:, 1])
        a.insert(len(a), area)

        for i in probas_[:, 1]:
            b.append(i)
    return a,b

def eval_panel(df, comb):
    for ff in comb:
        f = []
        for x in ff:
           f.insert(len(f),x)
        A,B = eval_features(df, f)
        #print("%s,%f,%s,%s" % (f, np.mean(A),A,B))
        check_best_models(A,f)
        print("%s,%f" % (f, np.mean(A)))
        file_features.write(str(f) + "\n")
        file_auc.write(str(np.mean(A)) + "\n")
        sys.stdout.flush()

def check_best_models(acc,features):
    global best_models, best_generated_model, feat

    model_accuracy = np.mean(acc)*100

    # check the number of models above the baseline model
    if (model_accuracy > 78.5):
        best_models = best_models + 1
        if (len(features) < len(feat)):
            feat = features
    # check the highestes model achieved
    if (model_accuracy > best_generated_model):
        best_generated_model = model_accuracy
        

# Reads dataset
df_mblood = pd.read_csv(sys.argv[1])

# Maps label
df_mblood.dropna(axis=0, subset=['satisfaction'], inplace=True)

#all_features = list(df_mblood.columns)
#for f in UNWANTED_COLUMNS + [LABEL_COLUMN_NAME]:
#    all_features.remove(f)

RANDOM_STATE = 1
f = []
i = 0
#for f1 in all_features:
for f1 in WANTED_COLUMNS:
    if i == 20: break
    if f1 in f: continue
    k = 0
    x = f1
    i = i + 1
    j = 0
    avg = 0
#    for f2 in all_features:
    for f2 in WANTED_COLUMNS:
         if f2 in f: continue
         j = j + 1
         f.insert(len(f), f2)
         A,B = eval_features(df_mblood, f)
         #print("%s,%f,%s,%s" % (f,np.mean(A),A,B))
         check_best_models(A,f)
         print("%s,%f" % (f,np.mean(A)))
         file_features.write(str(f) + "\n")
         file_auc.write(str(np.mean(A)) + "\n")
         f.remove(f2)
         sys.stdout.flush()
         avg = avg + np.mean(A)
         if np.mean(A) > k:
             x = f2
             k = np.mean(A)
    avg /= j
    f.insert(len(f), x)

for c in range(1,5):
    s = 50000
    comb = random_combinations(WANTED_COLUMNS, c, s)
    eval_panel(df_mblood, comb)

percentage = (best_models / total) * 100

with open('../reports/latin.txt', 'w') as f:
    print("Total number of models: %i\nBest achieved model: %f\nFeatures related to the smallest set of features: %s\nNumber of best models: %i \nPercentage of best models: %f" % (total,best_generated_model,feat,best_models,percentage), file=f)

file_features.close()
file_auc.close()

print('file written!!!')