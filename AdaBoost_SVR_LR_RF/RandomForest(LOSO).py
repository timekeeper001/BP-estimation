import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#data = pd.read_csv('/Users/jni/Documents/PyCode_FMP/data/data_47p_1h_21f.csv')
data=pd.read_csv('/Users/jni/Documents/PyCode_FMP/data/QingData_sbp.csv')
#data=data.sample(frac=0.02)
patient_list=data['patient_id'].unique()
#print(patient_list)
#print(data['patient_id'].value_counts())
feature=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
#feature=['HrRR','HrSS','AmFS','AmSN','AmFN','AmFN_FS','AmFN_SN','TmFN','TmNF','TmFN_NF','Tm_FS','Tm_SF','Tm_SN','Tm_FQ','PAT_S','PAT_F','PAT_Q','ArFS','ArSN','ArNF','ArNF_FN']

N=len(patient_list)

MAE_intersub=0
ME_intersub=0
STD_intersub=0

MAE_retrain=0
ME_retrain=0
STD_retrain=0

MAE_withinsub=0
ME_withinsub=0
STD_withinsub=0

retrain_size=0.05

error_vector=[]
retrainerror_vector=[]
truevalue_vector=[]
NN=0
for i in range(N):
    source_data=data[data['patient_id']!=patient_list[i]]
    target_data=data[data['patient_id']==patient_list[i]]
    retrain_data=target_data.iloc[0:int(retrain_size*len(target_data))]
    test_data=target_data.iloc[int(retrain_size*len(target_data)):len(target_data)]

    X_train=source_data[feature]#定义源域训练集
    y_train=source_data['SBP']
    X_retrain=retrain_data[feature]
    y_retrain=retrain_data['SBP']
    X_test=test_data[feature]
    y_test=test_data['SBP']

    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_retrain=scaler.transform(X_retrain)
    X_test=scaler.transform(X_test)



    model_rf=RandomForestRegressor(random_state=1)
    model_rf.fit(X_retrain,y_retrain)
    predictions_rf=model_rf.predict(X_test)
    mae_withinsub=mean_absolute_error(y_test, predictions_rf)
    MAE_withinsub+=mae_withinsub

    error_withinsub=predictions_rf-y_test
    me_withinsub=np.mean(error_withinsub)
    ME_withinsub+=me_withinsub
    std_withinsub=np.std(error_withinsub,ddof=1)
    STD_withinsub+=std_withinsub*(X_test.shape[0]-1)
    print(i,'within-subject training, MAE: ', mae_withinsub,'ME: ', me_withinsub, 'STD: ', std_withinsub)
    NN+=X_test.shape[0]-1
print('withinsub MAE: ', MAE_withinsub/N,'ME: ',ME_withinsub/N, 'STD: ', STD_withinsub/NN)

'''

    ####第一次训练

    model_rf=RandomForestRegressor(random_state=1,warm_start=True)#定义模型
    model_rf.fit(X_train,y_train)#初次训练
    predictions_rf=model_rf.predict(X_test)

    mae_intersub=mean_absolute_error(y_test, predictions_rf)
    MAE_intersub+=mae_intersub

    error_intersub=predictions_rf-y_test
    error_vector=np.append(error_vector,error_intersub)
    me_intersub=np.mean(error_intersub)
    ME_intersub+=me_intersub
    std_intersub=np.std(error_intersub,ddof=1)
    STD_intersub+=std_intersub*(X_test.shape[0]-1)
    print(i,'inter-subject training, MAE: ', mae_intersub,'ME: ', me_intersub, 'STD: ', std_intersub)
    #####重训练
    model_rf.n_estimators+=50
    model_rf.fit(X_retrain,y_retrain)
    predictions_rf2=model_rf.predict(X_test)
    
    mae_retrain=mean_absolute_error(y_test, predictions_rf2)
    MAE_retrain+=mae_retrain

    error_retrain=predictions_rf2 - y_test
    retrainerror_vector=np.append(retrainerror_vector,error_retrain)
    me_retrain=np.mean(error_retrain)
    ME_retrain+=me_retrain
    std_retrain=np.std(error_retrain,ddof=1)
    STD_retrain+=std_retrain*(X_test.shape[0]-1)
    print(i,'retrain, MAE: ',mae_retrain,'ME: ',me_retrain, 'STD: ',std_retrain)

    NN+=X_test.shape[0]-1
    truevalue_vector=np.append(truevalue_vector,y_test)


print(np.array(truevalue_vector).shape[0],np.array(error_vector).shape[0],np.array(retrainerror_vector).shape[0])
prediction_error={'SBP': truevalue_vector,
                 'prediction error': error_vector,
                 'prediction error after retrain': retrainerror_vector}
prediction_error=pd.DataFrame(prediction_error)
prediction_error.to_csv('SBP retrain RF.csv')###
print('inter-subject MAE: ', MAE_intersub/N, 'ME: ', ME_intersub/N, 'STD: ', STD_intersub/NN)
print('retrain MAE: ', MAE_retrain/N,'ME: ',ME_retrain/N, 'STD: ', STD_retrain/NN)

plt.figure(1)
plt.title("SBP of Random forest.csv") ###
plt.xlabel("true value of SBP") #
plt.ylabel("prediction error") 
plt.scatter(truevalue_vector,error_vector,s=0.2)  # 
plt.show()

plt.figure(2)
plt.title("SBP of Random forest after retrain.csv") ###
plt.xlabel("true value of SBP") #
plt.ylabel("prediction error") 
plt.scatter(truevalue_vector,retrainerror_vector,s=0.2,c='r')  # 
plt.show()
'''






