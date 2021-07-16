import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from TCA import kernel,TCA


data = pd.read_csv('data_20p_4h_19f.csv')
#data=pd.read_csv('QingData_sbp.csv')
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
F=5

X=data[feature]#定义源域训练集
y=data['SBP']
scaler=StandardScaler()
X=scaler.fit_transform(X)


groups=np.array(data['patient_id'])
gkf=GroupKFold(n_splits=F)
i=0
model=SVR()
print(model)
print('')
for train_index , val_index in gkf.split(X, y, groups):
    i+=1
    #print("TRAIN:", np.unique(groups[train_index]))
    print("TEST:", np.unique(groups[val_index]))
    X_train,y_train=X[train_index],y.iloc[train_index]
    X_val, y_val = X[val_index],y.iloc[val_index]
    tca=TCA(kernel_type='linear', dim=20, lamb=1, gamma=1)
    X_train, X_val=tca.fit(X_train, X_val)
    model.fit(X_train,y_train)
    fX_val=model.predict(X_val)

    # prediction_data={'ID':groups[val_index],
    #                  '真实值': y_val,
    #                  '预测值': fX_val }
    # prediction_data=pd.DataFrame(prediction_data) 
    # prediction_data.to_csv('fold '+ '%d' %i +'.csv')  

    mae=mean_absolute_error(y_val,fX_val)
    MAE+=mae
    error=fX_val-y_val
    me=np.mean(error)
    ME+=me
    std=np.std(error,ddof=1)
    STD+=std*(X_val.shape[0]-1)
    print('Validation on fold', i, 'MAE: ', format(mae,'.3f'),'ME: ', format(me,'.3f'), 'STD: ', format(std,'.3f'))
    print(' ')
print('Average MAE: ', format(MAE/F,'.3f'), 'ME: ', format(ME/F,'.3f'), 'STD: ', format(STD/(data.shape[0]-F),'.3f') )





'''
for i in range(3):
    source_data=data[data['patient_id']!=patient_list[i]]
    target_data=data[data['patient_id']==patient_list[i]]
    X=source_data[feature]#定义源域训练集
    y=source_data['SBP']
    X_test=target_data[feature]
    y_test=target_data['SBP']

    #按subject，划分训练&测试集，只划分一次
    # groups=np.array(source_data['patient_id'])
    # gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=i)
    # for train_index , val_index in gss.split(X, y, groups):
    #     print("TRAIN:", np.unique(groups[train_index]), "TEST:", np.unique(groups[val_index]))
    #     X_train,y_train=X.iloc[train_index],y.iloc[train_index]
    #     X_val, y_val = X.iloc[val_index],y.iloc[val_index]
    

    #训练集内部按subject进行cv,并超参数寻优
    
    param_grid = {'n_estimators': [150,200,250,300], 'learning_rate':[1,2,3,4]}
    min_mae=1000
    optimal_param={}
    for param in list(ParameterGrid(param_grid)):
        model=AdaBoostRegressor(random_state=1,n_estimators=param['n_estimators'],learning_rate=param['learning_rate'])#定义模型
        #print(model)
        gkf=GroupKFold(n_splits=4)
        mae_val=0
        for train_index , val_index in gkf.split(X, y, groups):
            #print("TRAIN:", np.unique(groups[train_index]), "TEST:", np.unique(groups[val_index]))
            X_train,y_train=X[train_index],y.iloc[train_index]
            X_val, y_val = X[val_index],y.iloc[val_index]

            model.fit(X_train,y_train)
            fX_val=model.predict(X_val)
            #print(mean_absolute_error(y_val, fX_val))
            mae_val+=mean_absolute_error(y_val, fX_val)
        mae_val=mae_val/4
        #print('mae: ', mae_val)
        if mae_val < min_mae:
            min_mae=mae_val
            optimal_param=param

    model=AdaBoostRegressor(random_state=1,n_estimators=optimal_param['n_estimators'],learning_rate=optimal_param['learning_rate'])
    print('optimal model: ', model)
    model.fit(X,y)
    fX_test=model.predict(X_test)
    mae=mean_absolute_error(y_test,fX_test)
    MAE+=mae
    error=y_test-fX_test
    me=np.mean(error)
    ME+=me
    std=np.std(error, ddof=1)
    STD+=std*(X_test.shape[0]-1)
    print('Testing on', i, 'MAE: ', mae,'ME: ', me, 'STD: ', std)
'''    



#print('within-subject MAE: ', MAE_withinsub/N, 'ME: ',ME_withinsub/N, 'STD: ', STD_withinsub/N)