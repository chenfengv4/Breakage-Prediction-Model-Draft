# Databricks notebook source
# MAGIC %md
# MAGIC ##PREDICT END STAGE OF TICKETS ALL AT ONCE: USED, REFUND, BREAKAGE

# COMMAND ----------

#document the variable definition and how each label is defined
#tc used/break/refund need to be clearly defined. if tc exchanged to ticket is called used for now.
##phase 2 need to track if tc -> exchanged to tkt -> not used
#ducument phase, model options, model predictions performance..
#phase 1 output more align with what accounting used for breakage

#phae 1 use case2 based on tkt of 2023 feb this is our prediction vs accounting

# COMMAND ----------

from murph import mosaic, rm_fileshare
group_name = 'group110'
fileshare = rm_fileshare(group_name)
mos = mosaic(tmode ='TERA', func_id = "MOSCPARA02", group_name = group_name)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

#import imblearn
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support



# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
stage1_input = mos.read_sql(
    """
WITH TEST AS (
SELECT
TICKET_NBR,
TICKET_ISSUE_MONTH,
FARE_BRAND_DESC,
TICKETING_CHANNEL,
TICKETING_POS,
FLIGHT_MONTH,
CURR_AADVAN_LEVEL_CD,
MAX_STATUS_OF_PNR,
TRIP_INTENT,
CUST_TRANSACTOR_T,
CUST_LOCATION_GRP,
TICKET_VALUE,
COUPON_VALUE_USD,
AP_IN_MONTHS,
TKT_LABEL,
STATUS_CHANGE_MONTH_BEFORE_FLT_MONTH,
END_STAGE_LABEL
FROM AA_RA_SANDBOX.TC_FC_TEST_2022_NOV_DEC_END_STAGE_LABEL
--WHERE TICKET_NBR = 0012360023835
)
SELECT
A.*,
B.INTEGERS AS SNPSHT_AP_IN_MONTHS,
CASE
WHEN B.INTEGERS > STATUS_CHANGE_MONTH_BEFORE_FLT_MONTH THEN 'OK-FLT-ATTACHED'
ELSE END_STAGE_LABEL
END AS SNPSHT_TKT_LABEL
FROM TEST A
JOIN AA_RA_SANDBOX.INTEGERS B
ON B.INTEGERS <= A.AP_IN_MONTHS AND B.INTEGERS > A.STATUS_CHANGE_MONTH_BEFORE_FLT_MONTH


"""
).toPandas()
#stage1_input.createOrReplaceTempView("stage1_input")

# COMMAND ----------

fileshare.to_csv(stage1_input, "ra/users/866810/Breakage_Model/test_data/end_stage_input.txt",sep='\t', index=False)

# COMMAND ----------

stage1_input.head(10)

# COMMAND ----------

stage1_input.SNPSHT_TKT_LABEL.value_counts()

# COMMAND ----------

stage1_input.shape

# COMMAND ----------

# MAGIC %md
# MAGIC #quick test random forest

# COMMAND ----------

input2 = fileshare.path(directory_name="ra/users/866810/Breakage_Model/test_data", file_name="end_stage_input.txt")
input2 = pd.read_csv(input2, sep='\t')
#input2.columns = input2.columns.str.lower()

# COMMAND ----------

input2.TKT_LABEL.value_counts()

# COMMAND ----------

input2.END_STAGE_LABEL.value_counts()

# COMMAND ----------

input2.SNPSHT_TKT_LABEL.value_counts()

# COMMAND ----------

input2.groupby(['TKT_LABEL'])['TICKET_VALUE'].sum()

# COMMAND ----------

input2.columns.tolist()

# COMMAND ----------

X=input2.drop(columns = [
'TICKET_NBR',
'COUPON_VALUE_USD',
'TKT_LABEL',
'STATUS_CHANGE_MONTH_BEFORE_FLT_MONTH',
#'SNPSHT_TKT_LABEL',
'END_STAGE_LABEL',
#'TICKET_VALUE',
#'TICKET_VALUE_BAND',
#'TICKET_ISSUE_MONTH',
#'FARE_BRAND_DESC',
#'TICKETING_CHANNEL',
#'TICKETING_POS',
#'FLIGHT_MONTH',
#'CURR_AADVAN_LEVEL_CD',
#'MAX_STATUS_OF_PNR',
#'TRIP_INTENT',
#'CUST_TRANSACTOR_T',
#'CUST_LOCATION_GRP'




 ])
y=input2.END_STAGE_LABEL
n_features = X.shape[1]

# COMMAND ----------

X.columns.tolist()

# COMMAND ----------

na_values = {"FARE_BRAND_DESC": 'Other', "TICKETING_CHANNEL": 'Other', "TICKETING_POS": 'OTHER', "CURR_AADVAN_LEVEL_CD": 'Other',"MAX_STATUS_OF_PNR": 'Other',"TRIP_INTENT": 'Other',"CUST_TRANSACTOR_T": 'Other',"CUST_LOCATION_GRP": 'Other','EMD_ISSUE_MONTH':'NA',"PROD_NM": 'Other',"RSN_FOR_ISSUANCE_SUB_CD": 'Other','FLIGHT_CREDIT_ISSUE_MONTH':'NA',"EMD_FLIGHT_MONTH_DIFF": 99,"FLIGHT_CREDIT_FLIGHT_MONTH_DIFF": 99,"EMD_USED_INTERVAL": 13,"FLIGHT_CREDIT_USED_INTERVAL": 13,
 'STATUS_CHANGE_MONTH_AFTER_FLT_MONTH':14,   'STATUS_CHANGE_MONTH_BEFORE_FLT_MONTH':14,'TICKET_VALUE':0}
X = X.fillna(value=na_values)

# COMMAND ----------

X_one_hot = pd.get_dummies(X, columns = [
'TICKET_ISSUE_MONTH',
'FARE_BRAND_DESC',
'TICKETING_CHANNEL',
'TICKETING_POS',
'FLIGHT_MONTH',
'CURR_AADVAN_LEVEL_CD',
'MAX_STATUS_OF_PNR',
'TRIP_INTENT',
'CUST_TRANSACTOR_T',
'CUST_LOCATION_GRP',
'SNPSHT_TKT_LABEL'
#'TICKET_VALUE_BAND'
  ])
#print(model_data_one_hot)
print(X_one_hot.columns.tolist())

X_one_hot.shape

# COMMAND ----------

X_one_hot.dropna()

# COMMAND ----------

y.shape

# COMMAND ----------

X.shape

# COMMAND ----------

X.head(10)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X_one_hot, y, random_state = 0)

# COMMAND ----------

# build RF
#build a forest of 100 trees
forest = RandomForestClassifier(n_estimators=100,random_state=0, max_depth=15)
forest_model = forest.fit(X_train, y_train)
RF_feature_name = X.columns
RF_class_name =forest.classes_

#took 4 minutes to finish the model building


# COMMAND ----------

#generate predictions
RF_predictions = forest_model.predict(X_test)
RF_pred_prob = forest_model.predict_proba(X_test)
#calculate performance
RF_predict_prob = (np.sum(RF_pred_prob, axis= 0)/len(y_test))
RF_predict_prob_df = pd.DataFrame((RF_predict_prob), index = RF_class_name)
RF_true_prob_df = pd.DataFrame(y_test.value_counts()/len(y_test))
RF_result = pd.concat([RF_predict_prob_df, RF_true_prob_df], axis=1).reindex(RF_predict_prob_df.index)
RF_result.columns = ['Predicted Prob', 'True Prob']
RF_result['Absolute Bias'] =abs( (RF_result['Predicted Prob'] - RF_result['True Prob'] )/RF_result['True Prob'])
RF_fitting_predictions = forest_model.predict(X_train)
print('Label based TRAINING accuracy',accuracy_score(y_train, RF_fitting_predictions) )
print('Label based PREDICTING accuracy is', accuracy_score(y_test, RF_predictions))
print('Aggregated Probability based PREDICTING accuracy is\n', RF_result)

# COMMAND ----------

precision, recall, fscore, support = precision_recall_fscore_support(y_test, RF_predictions)
print('category' + RF_class_name)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))

# COMMAND ----------

cm = confusion_matrix(y_test, RF_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=forest_model.classes_)
disp.plot()

# COMMAND ----------

#generate feature importance from RF
feature_importances = pd.DataFrame(forest_model.feature_importances_, index=X_one_hot.columns, columns=["Importance"])
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
feature_importances[:20].plot(kind='bar', figsize=(8,6))

# COMMAND ----------

single_tree_model = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)
#plot the tree
feature_name = X_one_hot.columns
class_name =single_tree_model.classes_
fig = plt.figure(figsize=(35,35))
_ = tree.plot_tree(single_tree_model, 
                   feature_names=feature_name,  
                   class_names=class_name,
                   filled=True,
                   fontsize = 15)

# COMMAND ----------

# MAGIC %md
# MAGIC #test multinomial logistic regression

# COMMAND ----------

input3 = fileshare.path(directory_name="ra/users/866810/Breakage_Model/test_data", file_name="end_stage_input.txt")
input3 = pd.read_csv(input3, sep='\t')
#input3.columns = input3.columns.str.lower()

# COMMAND ----------

input3.columns.tolist()

# COMMAND ----------

X3=input3.drop(columns = [
'TICKET_NBR',
'COUPON_VALUE_USD',
'TKT_LABEL',
'STATUS_CHANGE_MONTH_BEFORE_FLT_MONTH',
#'SNPSHT_TKT_LABEL',
#'TICKET_VALUE',

'TICKET_ISSUE_MONTH',
'FARE_BRAND_DESC',
'TICKETING_CHANNEL',
 'TICKETING_POS',
 'FLIGHT_MONTH',
 'CURR_AADVAN_LEVEL_CD',
 'MAX_STATUS_OF_PNR',
 'TRIP_INTENT',
# 'CUST_TRANSACTOR_T',
 'CUST_LOCATION_GRP',
 'END_STAGE_LABEL'
 ])
y3=input3.END_STAGE_LABEL
n_features3 = X3.shape[1]

# COMMAND ----------

X3.columns.tolist()

# COMMAND ----------

na_values = {"FARE_BRAND_DESC": 'Other', "TICKETING_CHANNEL": 'Other', "TICKETING_POS": 'OTHER', "CURR_AADVAN_LEVEL_CD": 'Other',"MAX_STATUS_OF_PNR": 'Other',"TRIP_INTENT": 'Other',"CUST_TRANSACTOR_T": 'Other',"CUST_LOCATION_GRP": 'Other','EMD_ISSUE_MONTH':'NA',"PROD_NM": 'Other',"RSN_FOR_ISSUANCE_SUB_CD": 'Other','FLIGHT_CREDIT_ISSUE_MONTH':'NA',"EMD_FLIGHT_MONTH_DIFF": 99,"FLIGHT_CREDIT_FLIGHT_MONTH_DIFF": 99,"EMD_USED_INTERVAL": 13,"FLIGHT_CREDIT_USED_INTERVAL": 13,
 'STATUS_CHANGE_MONTH_AFTER_FLT_MONTH':14,   'STATUS_CHANGE_MONTH_BEFORE_FLT_MONTH':14,'TICKET_VALUE':0}
X3 = X3.fillna(value=na_values)

# COMMAND ----------

X3_one_hot = pd.get_dummies(X3, columns = [
#'TICKET_ISSUE_MONTH',
#'FARE_BRAND_DESC',
#'TICKETING_CHANNEL',
# 'TICKETING_POS',
# 'FLIGHT_MONTH',
# 'CURR_AADVAN_LEVEL_CD',
#'MAX_STATUS_OF_PNR',
'SNPSHT_TKT_LABEL',
#'TICKET_VALUE_BAND',
#'TRIP_INTENT',
'CUST_TRANSACTOR_T'
# 'CUST_LOCATION_GRP',
  ])
#print(model_data_one_hot)
print(X3_one_hot.columns.tolist())

X3_one_hot.shape

# COMMAND ----------

X3_one_hot.columns.tolist()

# COMMAND ----------

y3.shape

# COMMAND ----------

X3_one_hot.dropna()

# COMMAND ----------

X3_train, X3_test, y3_train, y3_test = train_test_split(X3_one_hot, y3, random_state = 0)

# COMMAND ----------

X3_one_hot=X3_one_hot.drop(columns = ['CUST_TRANSACTOR_T_Other','CUST_TRANSACTOR_T_4_CURR_EARN_ONLY_OR_NO_EARN'])

# COMMAND ----------

import statsmodels.api as sm
X3_one_hot.constant = 1
logit_model=sm.MNLogit(y3,X3_one_hot)
result=logit_model.fit()
print(result.summary())
