import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.testing import all_estimators
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import warnings

"""""""""""""""
Variable
"""""""""""""""
df_Model_acc              = []
df_Model_acc              = pd.DataFrame(df_Model_acc)
df_Model_acc_temp         = pd.DataFrame(np.arange(2).reshape(1, 2))
df_Model_acc_temp.columns = ["Model", "accuracy_score"]

best_model_acc            = 100
best_model                = []

"""""""""""""""
Function
"""""""""""""""
def replace_to_number(data_frame_xxx):
    
    #Street
    column_mane = "Street"
    #column_element =[]
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Pave","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Grvl","0")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")
    
    #Alley
    column_mane = "Alley"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Pave","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Grvl","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")
    
    #ExterQual
    column_mane = "ExterQual"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Po","0")    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")
    
    #ExterCond
    column_mane = "ExterCond"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Po","0")    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")   

    #BsmtQual
    column_mane = "BsmtQual"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","5")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Po","1")  
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")   

    #BsmtCond
    column_mane = "BsmtCond"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","5")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Po","1")  
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")   

    #BsmtFinType1
    column_mane = "BsmtFinType1"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("GLQ","6")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("ALQ","5")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("BLQ","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Rec","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("LwQ","2")  
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Unf","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")   

    #BsmtFinType2
    column_mane = "BsmtFinType2"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("GLQ","6")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("ALQ","5")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("BLQ","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Rec","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("LwQ","2")  
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Unf","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")   
    
    #HeatingQC
    column_mane = "HeatingQC"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Po","0")    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")       

    #CentralAir
    column_mane = "CentralAir"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Y","0")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("N","1")   
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")     

    #KitchenQual
    column_mane = "KitchenQual"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Po","0")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8") 

    #FireplaceQu
    column_mane = "FireplaceQu"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","5")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Po","1")  
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")         

    #GarageQual
    column_mane = "GarageQual"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","5")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Po","1")  
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")    

    #GarageCond
    column_mane = "GarageCond"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","5")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Po","1")  
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")    

    #PavedDrive
    column_mane = "PavedDrive"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Y","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("P","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("N","0")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8")   

    #PoolQC
    column_mane = "PoolQC"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Ex","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Gd","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("TA","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("Fa","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8") 

    #Fence
    column_mane = "Fence"
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("GdPrv","4")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("MnPrv","3")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("GdWo","2")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].replace("MnWw","1")
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].fillna(0)    
    data_frame_xxx[column_mane] = data_frame_xxx[column_mane].astype("int8") 
        
    return data_frame_xxx



"""""""""""""""
Training
"""""""""""""""
#read csv
house_pr = pd.read_csv("./data/train.csv")

replace_to_number(house_pr)

#Data cleansing
house_pr_cl = house_pr.loc[:,house_pr.dtypes != object]
house_pr_cl = house_pr_cl.dropna()

#Preparation for traning data
y = house_pr_cl.loc[:,"SalePrice"]
x = house_pr_cl.drop("SalePrice", axis=1)

#Train/test data separation
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size = 0.2, random_state = 19)

#ver.6 run all algo
warnings.filterwarnings('ignore') 
allAlgorithms = all_estimators(type_filter = 'regressor')


for(name, algorithm) in allAlgorithms:
    
    #Select algorizm
    clf = algorithm()

    #Set algo name
    df_Model_acc_temp["Model"] = name

    #Exception
    try :
        #machine learning
        clf.fit(train_x, train_y)
        #prediction for the test data in train
        pred_y = clf.predict(test_x)
        #confirm performance of classifier
        RMSLE = np.sqrt(np.mean(((np.log(test_y+1)-np.log(pred_y+1))**2)))
        
        
        #serch best model
        if best_model_acc > RMSLE:
            best_model_acc = RMSLE
            best_model = algorithm()
            
            
    except : 
        df_Model_acc_temp["accuracy_score"] = 0
        continue
    else :
        df_Model_acc_temp["accuracy_score"] = RMSLE

    df_Model_acc = df_Model_acc.append(df_Model_acc_temp)

#Output accuracy
df_Model_acc = df_Model_acc.sort_values("accuracy_score", ascending=True).reset_index(drop=True)
df_Model_acc.to_csv("acc_score.csv", index=False)


"""""""""""""""
Set best regressor
"""""""""""""""

#Best Model re-learning
clf = best_model        
clf.fit(x, y)


"""""""""""""""
Validation
"""""""""""""""

#read csv
house_pr_test = pd.read_csv("./data/test.csv")

replace_to_number(house_pr_test)


#Data cleansing
house_pr_test_cl = house_pr_test.loc[:,house_pr.dtypes != object]
house_pr_test_cl = house_pr_test_cl.fillna(0)

#prediction for the test data in train
answer = clf.predict(house_pr_test_cl)

#reshape for output
Id = np.arange(1461,2920)
test = pd.DataFrame({"Id":Id, "SalePrice":answer})
test.to_csv("answer.csv", index=False)
