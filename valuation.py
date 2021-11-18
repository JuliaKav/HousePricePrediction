import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

boston_data=load_boston()

cols=boston_data['feature_names']
data=pd.DataFrame(data=boston_data['data'],columns=cols)
features = data.drop(['INDUS', 'AGE'], axis=1)
target=pd.DataFrame(np.log(boston_data['target']),columns=['PRICE'])


property_stats=np.ndarray(shape=(1,11))
property_stats=features.mean().values.reshape(1,11)
property_stats

regr=LinearRegression()
regr.fit(features,target)
fitted_vals=regr.predict(features)
MSE=mean_squared_error(target,fitted_vals)
RMSE=np.sqrt(MSE)

def get_log_estimate(noRooms,students_per_class,next_to_river=False,high_confidence=True):
    property_stats[0][4]=noRooms
    property_stats[0][8]=students_per_class
    if (next_to_river):
        property_stats[0][2]=1
    else:
        property_stats[0][2]=0
    log_estimate=regr.predict(property_stats)[0][0]
    
    if (high_confidence):
        upper=log_estimate + 2*RMSE
        lower=log_estimate - 2*RMSE
        interval=95
    else:
        upper=log_estimate + RMSE
        lower=log_estimate - RMSE
        interval=68
    return log_estimate,upper,lower,interval

def get_dollar_estimate(noRooms,students_per_class,next_to_river=False,high_confidence=True):
    ZILLOW_MEDIAN_PRICE = 583.3
    SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_data.target)
                        
    log_est, upper,lower,interval=get_log_estimate(noRooms,students_per_class,next_to_river, 
                                                   high_confidence)
    dol_est=np.round(np.e**log_est * SCALE_FACTOR * 1000 ,3)
    dol_up=np.round(np.e**upper * SCALE_FACTOR * 1000 ,3)                  
    dol_low=np.round(np.e**lower * SCALE_FACTOR * 1000, 3)
    return dol_est,dol_up,dol_low,interval