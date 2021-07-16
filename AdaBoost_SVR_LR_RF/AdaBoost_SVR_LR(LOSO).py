import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('/Users/jni/Documents/PyCode_FMP/data/data_47p_1h_21f.csv')
#data=pd.read_csv('/Users/jni/Documents/PyCode_FMP/data/QingData_sbp.csv')
patient_list=data['patient_id'].unique()
#print(patient_list)
#print(data['patient_id'].value_counts())
#feature=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
feature=['HrRR','HrSS','AmFS','AmSN','AmFN','AmFN_FS','AmFN_SN','TmFN','TmNF','TmFN_NF','Tm_FS','Tm_SF','Tm_SN','PAT_S','PAT_F','ArFS','ArSN','ArNF','ArNF_FN']

MAE=0
ME=0
STD=0
retrain_size=0.05
N=len(patient_list)
error_vector=[]
for i in range(N):
    source_data=data[data['patient_id']!=patient_list[i]]
    target_data=data[data['patient_id']==patient_list[i]]
    X_train=source_data[feature]#定义源域训练集
    y_train=source_data['SBP']
    X_test=target_data[feature]
    y_test=target_data['SBP']

    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    
    model=AdaBoostRegressor(random_state=1)
    model.fit(X_train,y_train)#初次训练
    y_predictions=model.predict(X_test)

    mae=mean_absolute_error(y_test, y_predictions)
    MAE+=mae
    error=y_predictions-y_test
    error_vector=np.append(error_vector,error)
    me=np.mean(error)
    ME+=me
    std=np.std(error,ddof=1)
    STD+=std*(X_test.shape[0]-1)
    print(i,'inter-subject training, MAE: ', mae,'ME: ', me, 'STD: ', std)
    
    #只用该患者的前5%数据训练，然后在剩余数据上预测。
    '''
    retrain_data=target_data.iloc[0:int(retrain_size*len(target_data))]
    test2_data=target_data.iloc[int(retrain_size*len(target_data)):len(target_data)]
    X_retrain=retrain_data[feature]
    y_retrain=retrain_data['SBP']
    X_test2=test2_data[feature]
    y_test2=test2_data['SBP']

    model2.fit(X_retrain,y_retrain)
    y_predictions2=model2.predict(X_test2)
    mae_withinsub=mean_absolute_error(y_test2, y_predictions2)
    MAE_withinsub+=mae_withinsub

    error_withinsub=y_predictions2-y_test2
    me_withinsub=np.mean(error_withinsub)
    ME_withinsub+=me_withinsub
    std_withinsub=np.std(error_withinsub)
    STD_withinsub+=std_withinsub
    print(i,'within-subject training, MAE: ',mae_withinsub,'ME: ',me_withinsub, 'STD: ',std_withinsub)
    print('  ')
    '''

prediction_error={'SBP': np.array(data['SBP']),
                 'prediction error': error_vector}
prediction_error=pd.DataFrame(prediction_error)
prediction_error.to_csv('SBP Random Forest.csv')###
plt.title("SBP of Random Forest") ###
plt.xlabel("true value of SBP") #
plt.ylabel("prediction error") 
plt.scatter(np.array(data['SBP']),error_vector.transpose(),s=0.2)  #               
print('Average MAE: ', MAE/N, 'ME: ', ME/N, 'STD: ', STD/(data.shape[0]-N) )
plt.show()
#print('within-subject MAE: ', MAE_withinsub/N, 'ME: ',ME_withinsub/N, 'STD: ', STD_withinsub/N)