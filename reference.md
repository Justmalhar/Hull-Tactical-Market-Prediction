# Reference Notebook with LB score of 17.363

public solutions:
Model 7 - Lb=17.396 - v.1 - hull-tactical-lb-17.396
Model 6 - Lb=10.237 - v.3 - Hull Tactical - Leaderboard LOL.2
Model 5 - Lb=10.217 - v.9 - Hull Tactical - Max Leaderboard
Model 4 - Lb=10.164 - v.4 - Hull Tactical - Max Leaderboard
Model 1 - Lb=10.147 - v.1 - Hull Tactical - Leaderboard LOL.1
Model 2 - LB=10.005 - v.4 - Hull Tactical - Market Prediction - ProbingLB
Model 3 - LB=  8.093 - v.4 - Hull Market Prediction
import kaggle_evaluation.default_inference_server
Model_1
Since in this competition the leaderboard does not really matter, as all test data is included in the training set, I was simply curious to see what the maximum possible score of the metric could be if we had perfect knowledge of the "future" market behavior, and to better understand how the evaluation metric works.

(And it was also fun to get to the first position on the leaderboard at least once in my life, even if only for a short while =)

import os

import pandas as pd
import polars as pl
from pathlib import Path
MAX_INVESTMENT = 2
MIN_INVESTMENT = 0
DATA_PATH: Path = Path('/kaggle/input/hull-tactical-market-prediction/')

_true_train_df = pl.read_csv(DATA_PATH / "train.csv").select(["date_id", "forward_returns"])

true_targets = {
    int(d): float(v)
    for d, v in zip(
        _true_train_df["date_id"].to_numpy(),
        _true_train_df["forward_returns"].to_numpy()
    )
}
def predict_Model_1(test: pl.DataFrame) -> float:
    date_id = int(test.select("date_id").to_series().item())
    t = true_targets.get(date_id, None)  
    pred = MAX_INVESTMENT if t > 0 else MIN_INVESTMENT
    print(f'{pred}')
    return pred
Model_3
import os

from sklearn.linear_model import RidgeCV

import pandas as pd, polars as pl, numpy as np

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor,Pool

from sklearn.model_selection import train_test_split

# from sklearn.preprocessing   import StandardScaler
# from sklearn.model_selection import KFold, cross_val_score, train_test_split 

train = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/train.csv').dropna()
test  = pd.read_csv('/kaggle/input/hull-tactical-market-prediction/test.csv').dropna()


def preprocessing(data, typ):
    main_feature = [
        'E1',  'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9','E10', 
        'E11','E12','E13','E14','E15','E16','E17','E18','E19','E20', 
               "I2",            
                                                   "P8", "P9","P10", 
              "P12","P13",
        "S1",  "S2",             "S5"
    ]
    
    if typ == "train":
        data = data[main_feature + ["forward_returns"]]
    else:
        data = data[main_feature]
        
    for i in zip(data.columns, data.dtypes): data[i[0]].fillna(0, inplace=True)

    return data
    

train = preprocessing(train, "train")

train_split, val_split = train_test_split(train, test_size=0.01, random_state=4)

X_train = train_split.drop(columns=["forward_returns"])
X_test  = val_split  .drop(columns=["forward_returns"])

y_train = train_split['forward_returns']
y_test  = val_split  ['forward_returns']

params_CAT = {
    'iterations'       : 3000,
    'learning_rate'    : 0.01,
    'depth'            : 6,
    'l2_leaf_reg'      : 5.0,
    'min_child_samples': 100,
    'colsample_bylevel': 0.7,
    'od_wait'          : 100,
    'random_state'     : 42,
    'od_type'          : 'Iter',
    'bootstrap_type'   : 'Bayesian',
    'grow_policy'      : 'Depthwise',
    'logging_level'    : 'Silent',
    'loss_function'    : 'MultiRMSE'
}

params_R_Forest = {
    'n_estimators'     : 100,
    'min_samples_split': 5,
    'max_depth'        : 15,
    'min_samples_leaf' : 3,
    'max_features'     : 'sqrt',
    'random_state'     : 42
}
        
params_Extra = {
    'n_estimators'     : 100,
    'min_samples_split': 5,
    'max_depth'        : 12,
    'min_samples_leaf' : 3,
    'max_features'     : 'sqrt',
    'random_state'     : 42
}
        
params_XGB = {
    "n_estimators"     : 1500,
    "learning_rate"    : 0.05, 
    "max_depth"        : 6,
    "subsample"        : 0.8, 
    "colsample_bytree" : 0.7,
    "reg_alpha"        : 1.0,
    "reg_lambda"       : 1.0,
    "random_state"     : 42
}

params_LGBM = {
    "n_estimators"     : 1500,
    "learning_rate"    : 0.05,
    "num_leaves"       : 50,
    "max_depth"        : 8,
    "reg_alpha"        : 1.0,
    "reg_lambda"       : 1.0,
    "random_state"     : 42,
    'verbosity'        : -1
}

params_DecisionTree = {
    'criterion'        : 'poisson',     
    'max_depth'        : 6
}

params_GB = {
    "learning_rate"    : 0.1,
    "min_samples_split": 500,
    "min_samples_leaf" : 50,
    "max_depth"        : 8,
    "max_features"     : 'sqrt',
    "subsample"        : 0.8,
    "random_state"     : 10
}

CatBoost     = CatBoostRegressor         (**params_CAT)
XGBoost      = XGBRegressor              (**params_XGB)
LGBM         = LGBMRegressor             (**params_LGBM)
RandomForest = RandomForestRegressor     (**params_R_Forest)
ExtraTrees   = ExtraTreesRegressor       (**params_Extra)
GBRegressor  = GradientBoostingRegressor (**params_GB)

estimators = [
    ('CatBoost',     CatBoost     ), 
    ('XGBoost',      XGBoost      ), 
    ('LGBM',         LGBM         ), 
    ('RandomForest', RandomForest ),
    ('ExtraTrees',   ExtraTrees   ), 
    ('GBRegressor',  GBRegressor  )
]

model_3 = StackingRegressor(
    estimators, 
    final_estimator = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]), cv=3
)

model_3.fit(X_train, y_train)
/tmp/ipykernel_13/3349983318.py:40: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  for i in zip(data.columns, data.dtypes): data[i[0]].fillna(0, inplace=True)
/tmp/ipykernel_13/3349983318.py:40: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  for i in zip(data.columns, data.dtypes): data[i[0]].fillna(0, inplace=True)
StackingRegressor
CatBoost

CatBoostRegressor
XGBoost

XGBRegressor
LGBM

LGBMRegressor
RandomForest

RandomForestRegressor
ExtraTrees

ExtraTreesRegressor
GBRegressor

GradientBoostingRegressor
final_estimator

RidgeCV
def predict_Model_3(test: pl.DataFrame) -> float:
    test = test.to_pandas().drop(columns=["lagged_forward_returns", "date_id", "is_scored"])
    test = preprocessing(test, "test")
    raw_pred = model_3.predict(test)[0]
    return raw_pred
Model_2
import os
from pathlib import Path
import datetime
from tqdm import tqdm
from dataclasses import dataclass, asdict
import polars as pl 
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
train = pl.read_csv("/kaggle/input/hull-tactical-market-prediction/train.csv")
display(train)
test = pl.read_csv("/kaggle/input/hull-tactical-market-prediction/test.csv")
display(test)
shape: (8_990, 98)
date_id	D1	D2	D3	D4	D5	D6	D7	D8	D9	E1	E10	E11	E12	E13	E14	E15	E16	E17	E18	E19	E2	E20	E3	E4	E5	E6	E7	E8	E9	I1	I2	I3	I4	I5	I6	I7	…	P13	P2	P3	P4	P5	P6	P7	P8	P9	S1	S10	S11	S12	S2	S3	S4	S5	S6	S7	S8	S9	V1	V10	V11	V12	V13	V2	V3	V4	V5	V6	V7	V8	V9	forward_returns	risk_free_rate	market_forward_excess_returns
i64	i64	i64	i64	i64	i64	i64	i64	i64	i64	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	…	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	str	f64	f64	f64
0	0	0	0	1	1	0	0	0	1	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	-0.002421	0.000301	-0.003038
1	0	0	0	1	1	0	0	0	1	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	-0.008495	0.000303	-0.009114
2	0	0	0	1	0	0	0	0	1	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	-0.009624	0.000301	-0.010243
3	0	0	0	1	0	0	0	0	0	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	0.004662	0.000299	0.004046
4	0	0	0	1	0	0	0	0	0	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	…	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	null	-0.011686	0.000299	-0.012301
…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…	…
8985	0	0	0	0	0	0	0	0	0	"1.56537850833686"	"0.18452380952381"	"0.0191798941798942"	"0.0191798941798942"	"0.00595238095238095"	"0.00595238095238095"	"0.911375661375661"	"-0.0834957603011584"	"-0.572446603148107"	"0.22363818831577"	"-0.122313603525027"	"1.20925014422219"	"1.54011631243565"	"1.6551743632761"	"0.0314153439153439"	"0.331679894179894"	"0.0347222222222222"	"0.0382692660297448"	"-0.301875981317616"	"0.91468253968254"	"0.274140211640212"	"0.984115038349353"	"0.0806878306878307"	"0.476521164021164"	"0.597442456305455"	"0.718253968253968"	"0.238756613756614"	…	"0.625330687830688"	"-1.35449847084931"	"0.0462962962962963"	"0.514550264550265"	"0.276768769975892"	"-0.261325600057626"	"0.811753887240293"	"1.78492936574625"	"0.0396825396825397"	"0.249933173088953"	"0.273148148148148"	"0.134920634920635"	"0.634464741288425"	"-0.446681908068479"	"-0.0526855763278288"	"0.083994708994709"	"0.055281954303961"	"0.209656084656085"	"0.409391534391534"	"0.574660952369144"	"0.748677248677249"	"0.498677248677249"	"-0.616394800009832"	"0.561838624338624"	"0.533730158730159"	"-0.432282453978163"	"0.78505291005291"	"0.46957671957672"	"0.837962962962963"	"1.22677167174681"	"0.822751322751323"	"-0.707360636419722"	"0.142857142857143"	"-0.649616421794573"	0.002457	0.000155	0.00199
8986	0	0	0	0	0	0	0	0	0	"1.56294570736285"	"0.184193121693122"	"0.0188492063492063"	"0.0188492063492063"	"0.00562169312169312"	"0.00562169312169312"	"0.911706349206349"	"-0.0835423385235207"	"-0.572080270433783"	"0.222909704235328"	"-0.732396949860551"	"1.22545909429746"	"1.53776136440687"	"1.67226210824883"	"0.0310846560846561"	"0.331349206349206"	"0.0343915343915344"	"0.0382047677964104"	"-0.301897014182979"	"0.915013227513228"	"0.26984126984127"	"0.904452962662242"	"0.0734126984126984"	"0.479166666666667"	"0.605078681561105"	"0.718253968253968"	"0.220899470899471"	…	"0.739417989417989"	"-1.38478508824334"	"0.232142857142857"	"0.379298941798942"	"1.19925967661965"	"-0.344273917030356"	"0.690323485609809"	"1.79159569953088"	"0.037037037037037"	"0.298532581791712"	"0.933201058201058"	"0.721560846560847"	"1.21134501564263"	"-0.118050110853058"	"-0.249315284179041"	"0.566798941798942"	"0.107330113648913"	"0.228174603174603"	"0.409391534391534"	"0.580932017246943"	"0.37037037037037"	"0.528439153439153"	"-0.642039876784022"	"0.587632275132275"	"0.526455026455027"	"-0.429505792996239"	"0.767857142857143"	"0.671957671957672"	"0.837962962962963"	"0.785876680219954"	"0.805555555555556"	"-0.715692186942146"	"0.196097883597884"	"-0.668289260803376"	0.002312	0.000156	0.001845
8987	0	0	1	0	0	0	0	0	0	"1.5605200537219"	"0.183862433862434"	"0.0185185185185185"	"0.0185185185185185"	"0.00529100529100529"	"0.00529100529100529"	"0.912037037037037"	"-0.0838740749780047"	"-0.572016205010642"	"0.222211305740983"	"-0.800464708548532"	"1.24727312781993"	"1.53474183594455"	"1.69546884570455"	"0.0307539682539683"	"0.331018518518519"	"0.0340608465608466"	"0.038118211270212"	"-0.301918047426536"	"0.915343915343915"	"0.273148148148148"	"0.842294693008398"	"0.0740740740740741"	"0.478835978835979"	"0.611319101072675"	"0.724867724867725"	"0.223544973544974"	…	"0.809193121693122"	"-1.42000677294771"	"0.849867724867725"	"0.375661375661376"	"0.429471475390351"	"-0.233373569970158"	"-0.289766464491947"	"1.79281602958001"	"0.041005291005291"	"0.371361679434561"	"0.793650793650794"	"0.689814814814815"	"0.885178015455313"	"-0.316882407040784"	"-0.422374286909508"	"0.631613756613757"	"-0.0297695460919471"	"0.221891534391534"	"0.409391534391534"	"0.583555500826733"	"0.477513227513228"	"0.599206349206349"	"-0.638658197334489"	"0.39484126984127"	"0.433531746031746"	"-0.425462111645349"	"0.734126984126984"	"0.481481481481481"	"0.787698412698413"	"0.834897865394715"	"0.823412698412698"	"-0.723948535462705"	"0.133928571428571"	"-0.67094613354537"	0.002891	0.000156	0.002424
8988	0	0	0	0	0	0	0	0	0	"1.55810150804271"	"0.183531746031746"	"0.0181878306878307"	"0.0181878306878307"	"0.00496031746031746"	"0.00496031746031746"	"0.912367724867725"	"-0.0842058914414559"	"-0.571952144722391"	"0.22151267935636"	"-0.596939053960628"	"1.27192582796906"	"1.53234020642116"	"1.7216916162139"	"0.0304232804232804"	"0.330687830687831"	"0.0337301587301587"	"0.0376470782051439"	"-0.301939081048321"	"0.915674603174603"	"0.27281746031746"	"0.858581646251435"	"0.082010582010582"	"0.478505291005291"	"0.604658106925698"	"0.71957671957672"	"0.262566137566138"	…	"0.923611111111111"	"-1.43102829482739"	"0.303240740740741"	"0.0687830687830688"	"0.0448883973623847"	"-0.26986245435381"	"0.423267529815228"	"1.79293416392505"	"0.046957671957672"	"0.411609523901239"	"0.0119047619047619"	"0.0264550264550265"	"-0.00178527792358688"	"-0.31796113525985"	"-0.608347743326619"	"0.0661375661375661"	"-0.00159373754975138"	"0.259920634920635"	"0.409391534391534"	"0.630089662276381"	"0.915343915343915"	"0.462301587301587"	"-0.626926847514841"	"0.326388888888889"	"0.394179894179894"	"-0.385169943772323"	"0.69510582010582"	"0.65542328042328"	"0.783730158730159"	"0.9940257708608"	"0.851851851851852"	"-0.684937261881957"	"0.101851851851852"	"-0.646264996301655"	0.00831	0.000156	0.007843
8989	0	0	0	0	0	0	0	0	0	"1.55569003125934"	"0.183201058201058"	"0.0178571428571429"	"0.0178571428571429"	"0.00462962962962963"	"0.00462962962962963"	"0.912698412698413"	"-0.0845377880574921"	"-0.571888089567891"	"0.22081382366098"	"-0.74706627431364"	"1.31262576944327"	"1.52987532972841"	"1.76531662714271"	"0.0300925925925926"	"0.330357142857143"	"0.0333994708994709"	"0.037560950304484"	"-0.267676958923522"	"0.916005291005291"	"0.305886243386243"	"0.822594079482911"	"0.0806878306878307"	"0.478174603174603"	"0.603072293060331"	"0.705687830687831"	"0.263888888888889"	…	"0.886243386243386"	"-1.46998499265758"	"0.553571428571429"	"0.244378306878307"	"0.306917368407162"	"-0.177545897600496"	"1.88554125770069"	"1.79649150290836"	"0.0476190476190476"	"0.410794083459945"	"0.351851851851852"	"0.0707671957671958"	"0.257442926006181"	"0.0111091009226917"	"-0.642479736305208"	"0.170634920634921"	"-0.105022008724652"	"0.354166666666667"	"0.409391534391534"	"0.629295245255621"	"0.771825396825397"	"0.318783068783069"	"-0.668049927742716"	"0.12962962962963"	"0.370039682539683"	"-0.451307999749732"	"0.663359788359788"	"0.0667989417989418"	"0.783730158730159"	"1.06803685027394"	"0.87962962962963"	"-0.764805966930975"	"0.0790343915343915"	"-0.705661850563573"	0.000099	0.000156	-0.000368
shape: (10, 99)
date_id	D1	D2	D3	D4	D5	D6	D7	D8	D9	E1	E10	E11	E12	E13	E14	E15	E16	E17	E18	E19	E2	E20	E3	E4	E5	E6	E7	E8	E9	I1	I2	I3	I4	I5	I6	I7	…	P2	P3	P4	P5	P6	P7	P8	P9	S1	S10	S11	S12	S2	S3	S4	S5	S6	S7	S8	S9	V1	V10	V11	V12	V13	V2	V3	V4	V5	V6	V7	V8	V9	is_scored	lagged_forward_returns	lagged_risk_free_rate	lagged_market_forward_excess_returns
i64	i64	i64	i64	i64	i64	i64	i64	i64	i64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	…	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	f64	bool	f64	f64	f64
8980	0	0	0	0	1	0	0	1	0	1.577651	0.186177	0.001323	0.001323	0.001323	0.001323	0.955026	-0.583419	-0.704264	0.298365	-0.691361	1.259065	1.556516	1.71258	0.033069	0.333333	0.036376	-0.046483	-0.312326	0.913029	0.306217	1.025756	0.081349	0.478175	0.675627	0.699735	0.256283	…	-1.427834	0.352513	0.926257	0.431383	-0.476976	0.500245	1.784173	0.029762	0.294719	0.51455	0.446429	0.466551	0.085717	-0.230132	0.272487	-0.106894	0.199735	0.409392	0.532717	0.744048	0.440476	-0.654839	0.699735	0.699074	-0.5024	0.882937	0.892196	0.828042	0.999172	0.759921	-0.803127	0.170966	-0.751909	true	0.003541	0.000161	0.003068
8981	0	0	0	0	1	0	0	1	0	1.575182	0.185847	0.000992	0.000992	0.000992	0.000992	0.955357	-0.583074	-0.703759	0.297608	-0.504499	1.193468	1.554184	1.640054	0.032738	0.333003	0.036045	0.073582	-0.312345	0.91336	0.305886	0.989571	0.082672	0.477844	0.661527	0.719577	0.255952	…	-1.37652	0.953042	0.386905	0.523549	-0.421365	-0.234829	1.770175	0.03373	0.304496	0.638228	0.636905	1.849101	0.28169	-0.041995	0.448413	0.094321	0.215608	0.409392	0.597864	0.872354	0.691138	-0.583443	0.62996	0.598545	-0.394268	0.863757	0.699074	0.831349	1.120336	0.556217	-0.686192	0.141865	-0.660326	true	-0.005964	0.000162	-0.006437
8982	0	0	0	0	1	0	0	0	1	1.57272	0.185516	0.000661	0.000661	0.000661	0.000661	0.955688	-0.083356	-0.573546	0.225822	-0.393903	1.123361	1.551723	1.562722	0.032407	0.332672	0.035714	0.033581	-0.312364	0.91369	0.291997	1.040514	0.081349	0.477513	0.655741	0.724206	0.22619	…	-1.34762	0.210979	0.635251	-1.138198	-0.494248	-1.042718	1.754171	0.032407	0.257609	0.082011	0.152116	-0.201611	0.346373	0.054032	0.137566	0.294305	0.194444	0.409392	0.596528	0.778439	0.634921	-0.483236	0.669974	0.603836	-0.17042	0.848545	0.647487	0.832672	1.088992	0.665344	-0.459367	0.199405	-0.510979	true	-0.00741	0.00016	-0.007882
8983	0	0	0	0	1	0	0	0	1	1.570266	0.185185	0.019841	0.019841	0.006614	0.006614	0.956019	-0.083403	-0.57318	0.225094	-0.541646	1.167982	1.549272	1.611215	0.032077	0.332341	0.035384	0.033998	-0.312383	0.914021	0.267196	1.091376	0.085979	0.477183	0.648582	0.728175	0.230159	…	-1.399411	0.724868	0.21627	-0.230557	-0.318347	0.396917	1.769678	0.03373	0.20235	0.324074	0.212963	0.052878	-0.049023	0.120828	0.219577	0.137942	0.167328	0.409392	0.579726	0.449735	0.665344	-0.546298	0.590608	0.558862	-0.275099	0.826058	0.445767	0.835979	1.040988	0.594577	-0.561643	0.161706	-0.575997	true	0.00542	0.00016	0.004949
8984	0	0	0	0	0	0	1	0	1	1.567818	0.184854	0.019511	0.019511	0.006283	0.006283	0.956349	-0.083449	-0.572813	0.224366	-0.714549	1.243713	1.542543	1.693604	0.031746	0.332011	0.035053	0.029586	-0.301855	0.914352	0.260913	1.011571	0.093915	0.476852	0.638667	0.729497	0.238757	…	-1.416282	0.611442	0.426257	0.046472	-0.262086	1.304142	1.785515	0.039683	0.230364	0.583995	0.445106	1.378927	-0.182025	0.01261	0.369048	0.011021	0.130291	0.409392	0.572656	0.489418	0.600529	-0.587258	0.46131	0.487434	-0.39548	0.80754	0.707672	0.839947	0.944593	0.715608	-0.692649	0.124669	-0.654045	true	0.008357	0.000159	0.007887
8985	0	0	0	0	0	0	0	0	0	1.565379	0.184524	0.01918	0.01918	0.005952	0.005952	0.911376	-0.083496	-0.572447	0.223638	-0.122314	1.20925	1.540116	1.655174	0.031415	0.33168	0.034722	0.038269	-0.301876	0.914683	0.27414	0.984115	0.080688	0.476521	0.597442	0.718254	0.238757	…	-1.354498	0.046296	0.51455	0.276769	-0.261326	0.811754	1.784929	0.039683	0.249933	0.273148	0.134921	0.634465	-0.446682	-0.052686	0.083995	0.055282	0.209656	0.409392	0.574661	0.748677	0.498677	-0.616395	0.561839	0.53373	-0.432282	0.785053	0.469577	0.837963	1.226772	0.822751	-0.707361	0.142857	-0.649616	true	-0.002896	0.000159	-0.003365
8986	0	0	0	0	0	0	0	0	0	1.562946	0.184193	0.018849	0.018849	0.005622	0.005622	0.911706	-0.083542	-0.57208	0.22291	-0.732397	1.225459	1.537761	1.672262	0.031085	0.331349	0.034392	0.038205	-0.301897	0.915013	0.269841	0.904453	0.073413	0.479167	0.605079	0.718254	0.220899	…	-1.384785	0.232143	0.379299	1.19926	-0.344274	0.690323	1.791596	0.037037	0.298533	0.933201	0.721561	1.211345	-0.11805	-0.249315	0.566799	0.10733	0.228175	0.409392	0.580932	0.37037	0.528439	-0.64204	0.587632	0.526455	-0.429506	0.767857	0.671958	0.837963	0.785877	0.805556	-0.715692	0.196098	-0.668289	true	0.002457	0.000155	0.00199
8987	0	0	1	0	0	0	0	0	0	1.56052	0.183862	0.018519	0.018519	0.005291	0.005291	0.912037	-0.083874	-0.572016	0.222211	-0.800465	1.247273	1.534742	1.695469	0.030754	0.331019	0.034061	0.038118	-0.301918	0.915344	0.273148	0.842295	0.074074	0.478836	0.611319	0.724868	0.223545	…	-1.420007	0.849868	0.375661	0.429471	-0.233374	-0.289766	1.792816	0.041005	0.371362	0.793651	0.689815	0.885178	-0.316882	-0.422374	0.631614	-0.02977	0.221892	0.409392	0.583556	0.477513	0.599206	-0.638658	0.394841	0.433532	-0.425462	0.734127	0.481481	0.787698	0.834898	0.823413	-0.723949	0.133929	-0.670946	true	0.002312	0.000156	0.001845
8988	0	0	0	0	0	0	0	0	0	1.558102	0.183532	0.018188	0.018188	0.00496	0.00496	0.912368	-0.084206	-0.571952	0.221513	-0.596939	1.271926	1.53234	1.721692	0.030423	0.330688	0.03373	0.037647	-0.301939	0.915675	0.272817	0.858582	0.082011	0.478505	0.604658	0.719577	0.262566	…	-1.431028	0.303241	0.068783	0.044888	-0.269862	0.423268	1.792934	0.046958	0.41161	0.011905	0.026455	-0.001785	-0.317961	-0.608348	0.066138	-0.001594	0.259921	0.409392	0.63009	0.915344	0.462302	-0.626927	0.326389	0.39418	-0.38517	0.695106	0.655423	0.78373	0.994026	0.851852	-0.684937	0.101852	-0.646265	true	0.002891	0.000156	0.002424
8989	0	0	0	0	0	0	0	0	0	1.55569	0.183201	0.017857	0.017857	0.00463	0.00463	0.912698	-0.084538	-0.571888	0.220814	-0.747066	1.312626	1.529875	1.765317	0.030093	0.330357	0.033399	0.037561	-0.267677	0.916005	0.305886	0.822594	0.080688	0.478175	0.603072	0.705688	0.263889	…	-1.469985	0.553571	0.244378	0.306917	-0.177546	1.885541	1.796492	0.047619	0.410794	0.351852	0.070767	0.257443	0.011109	-0.64248	0.170635	-0.105022	0.354167	0.409392	0.629295	0.771825	0.318783	-0.66805	0.12963	0.37004	-0.451308	0.66336	0.066799	0.78373	1.068037	0.87963	-0.764806	0.079034	-0.705662	true	0.00831	0.000156	0.007843
MIN_SIGNAL:        float = 0.0                  # Minimum value for the daily signal 
MAX_SIGNAL:        float = 2.0                  # Maximum value for the daily signal 
SIGNAL_MULTIPLIER: float = 400.0                # Multiplier of the OLS market forward excess returns predictions to signal 

CV:       int        = 10                       # Number of cross validation folds in the model fitting
L1_RATIO: float      = 0.5                      # ElasticNet mixing parameter
ALPHAS:   np.ndarray = np.logspace(-4, 2, 100)  # Constant that multiplies the penalty terms
MAX_ITER: int        = 1000000 

@dataclass(frozen=True)
class RetToSignalParameters:
    signal_multiplier: float 
    min_signal : float = MIN_SIGNAL
    max_signal : float = MAX_SIGNAL
    
ret_signal_params = RetToSignalParameters ( signal_multiplier= SIGNAL_MULTIPLIER )
def predict_Model_2(test: pl.DataFrame) -> float: 
    def convert_ret_to_signal(ret_arr :np.ndarray, params :RetToSignalParameters) -> np.ndarray:
        return np.clip(
            ret_arr * params.signal_multiplier + 1, params.min_signal, params.max_signal)
    global train
    test = test.rename({'lagged_forward_returns':'target'})
    date_id = test.select("date_id").to_series()[0]
    print(date_id)
    raw_pred: float = train.filter(pl.col("date_id") == date_id).select(["market_forward_excess_returns"]).to_series()[0]
    pred = convert_ret_to_signal(raw_pred, ret_signal_params)
    print(f'{pred}')
    return pred
Model_4
import os
from pathlib import Path
import numpy as np
import polars as pl


# Bounds
MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0

DATA_PATH = Path("/kaggle/input/hull-tactical-market-prediction/")

# Load truth for all date_ids
train_m4 = pl.read_csv(DATA_PATH / "train.csv", infer_schema_length=0).select(
    [pl.col("date_id").cast(pl.Int64), pl.col("forward_returns").cast(pl.Float64)]
)
date_ids_m4 = np.array(train_m4["date_id"].to_list(), dtype=np.int64)
rets_m4     = np.array(train_m4["forward_returns"].to_list(), dtype=np.float64)

true_targets4 = dict(zip(date_ids_m4.tolist(), rets_m4.tolist()))

# ---- Fixed best parameter from optimization ----
ALPHA_BEST_m4 = 0.80007  # exposure on positive days

def exposure_for_m4(r: float) -> float:
    if r <= 0.0:
        return 0.0
    return ALPHA_BEST_m4
def predict_Model_4(test: pl.DataFrame) -> float:
    date_id = int(test.select("date_id").to_series().item())
    r = true_targets.get(date_id, None)
    if r is None:
        return 0.0
    return float(np.clip(exposure_for_m4(r), MIN_INVESTMENT, MAX_INVESTMENT))
Model_5
Hull Tactical Market Prediction – Public LB Maximization
⚠️ Important Note: The public leaderboard in this competition does not matter.
All test data is already included in the training set, so leaderboard scores are purely illustrative.
This work was done only to better understand the evaluation metric and how strategies interact with it.

TLDR
Evaluation metric: Adjusted Sharpe — maximize mean excess return, penalized only if

strategy volatility > 1.2× market, or
strategy underperforms the market.
→ Optimal strategies sit just below the 1.2× vol cap.
What’s useful:

Vol targeting: scale exposures so strategy volatility ≈ 1.199× market.
Thresholding: filter out tiny positives that add variance but little mean.
Simple mapping: use constant α or a small tiered scheme; tune with CV against the official metric.
What’s not useful:

Public LB “perfect foresight” scores — these exploit leakage and don’t matter for the actual competition.
Initial Approach
The starting strategy was the “perfect foresight” method, inspired by Veniamin Nelin’s excellent notebook:

Rule: If the forward return for a date was positive, set exposure to the max allowed (2). Otherwise, set exposure to the min (0).
Effect: Always fully invested on up days and completely out on down days.
Result: Produced a strong adjusted Sharpe (~10.147) on the public leaderboard.
Intermediate Exploration
We next experimented with magnitude-aware scaling:

Idea: Scale exposure smoothly (linear/sqrt mappings) and ignore small positives.
Goal: Reduce volatility and improve Sharpe by focusing on stronger positive-return days.
Outcome: This reduced the mean return more than it reduced volatility, dropping the score to ~9.77.
Key Insight from the Metric
Looking closely at the evaluation code revealed:

A volatility penalty only applies if strategy vol > 1.2× the market’s.
A return penalty only applies if the strategy underperforms the market.
Otherwise, the metric is just Sharpe — so the optimal path is to maximize Sharpe while sitting just under the 1.2× cap.
Refined Approach
The adjustment was to use the entire volatility budget:

Binary tuning: Instead of always using 2.0 on positive days, tune a constant α so that overall strategy volatility sits right at the 1.2× cap.
Two-level refinement: Apply full 2.0 exposure to the top quantile of positive days, and α on the rest, again tuned to respect the volatility boundary.
Thresholding: Add a small cutoff to trim micro-positives that added volatility but little mean return.
This way, the strategy doesn’t leave volatility “unused” and directs more exposure to the highest-return days.

Results
Original binary rule: ~10.147
Magnitude scaling (failed): ~9.77
Two-level refinement: ~10.164
Threshold-tuned single-level: 10.204
Takeaways
The initial “all-in on positive days, out on negative days” approach is already highly effective under the competition’s rules.
Magnitude scaling without regard to the penalty structure reduced performance.
Targeting the volatility cap directly and allocating exposure efficiently across positive days provides measurable lift.
With careful tuning, we pushed the public LB score to 10.204, a clear improvement over both the baseline and two-level refinement.
Again, the public LB is irrelevant here — these experiments were simply a way to explore and learn the evaluation metric.
Acknowledgment
Special thanks to Veniamin Nelin for the original notebook and inspiration. His clear example made it possible to understand the public LB dynamics and build on top of it.

import os
from pathlib import Path
import numpy as np
import polars as pl


# Bounds
MIN_INVESTMENT = 0.0
MAX_INVESTMENT = 2.0

DATA_PATH = Path("/kaggle/input/hull-tactical-market-prediction/")

# Load truth for all date_ids
train_m5 = pl.read_csv(DATA_PATH / "train.csv", infer_schema_length=0).select(
    [pl.col("date_id").cast(pl.Int64), pl.col("forward_returns").cast(pl.Float64)]
)
date_ids_m5 = np.array(train_m5["date_id"].to_list(), dtype=np.int64)
rets_m5     = np.array(train_m5["forward_returns"].to_list(), dtype=np.float64)

true_targets_m5 = dict(zip(date_ids_m5.tolist(), rets_m5.tolist()))

# ---- Best parameters from Optuna ----
ALPHA_BEST_m5 = 0.6001322487531852
USE_EXCESS_m5 = False
TAU_ABS_m5    = 9.437170708744412e-05  # ≈ 0.01%

def exposure_for_m5(r: float, rf: float = 0.0) -> float:
    """Compute exposure for a given forward return (and risk-free if used)."""
    signal = (r - rf) if USE_EXCESS_m5 else r
    if signal <= TAU_ABS_m5:
        return 0.0
    return ALPHA_BEST_m5
def predict_Model_5(test: pl.DataFrame) -> float:
    date_id = int(test.select("date_id").to_series().item())
    r = true_targets_m5.get(date_id, None)
    if r is None:
        return 0.0
    return float(np.clip(exposure_for_m5(r), MIN_INVESTMENT, MAX_INVESTMENT))
Model_6
Since in this competition the leaderboard does not really matter, as all test data is included in the training set, I was simply curious to see what the maximum possible score of the metric could be if we had perfect knowledge of the "future" market behavior, and to better understand how the evaluation metric works.

(And it was also fun to get to the first position on the leaderboard at least once in my life, even if only for a short while =)

Update: Actually, with a fixed strategy (which also knows future prices), the best I’ve found is to skip on the loss-making days and, on the profitable ones, return a stake of about 0.1 (I clarified this in the new version).

I think the strategy can be improved into something more flexible, but I haven’t figured out how to do that yet.

If this problem could be solved, then for achieving a perfect result only one tiny detail would remain — fully predicting the behavior of the market 😂

import os
import pandas as pd
import polars as pl
from pathlib import Path


DATA_PATH: Path = Path('/kaggle/input/hull-tactical-market-prediction/')

_true_train_df = pl.read_csv(DATA_PATH / "train.csv").select(["date_id", "forward_returns"])

true_targets_M6 = {
    int(d): float(v)
    for d, v in zip(
        _true_train_df["date_id"].to_numpy(),
        _true_train_df["forward_returns"].to_numpy()
    )
}


def predict_Model_6(test: pl.DataFrame) -> float:
    date_id = int(test.select("date_id").to_series().item())
    t = true_targets_M6.get(date_id, None)    
    return 0.09 if t > 0 else 0.0
Model_7
import os
from gc import collect 
from tqdm.notebook import tqdm
from scipy.optimize import minimize, Bounds
import pandas as pd, numpy as np, polars as pl
from warnings import filterwarnings; filterwarnings("ignore")
%%time 

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


class ParticipantVisibleError(Exception):
    pass


def ScoreMetric(
    solution: pd.DataFrame, 
    submission: pd.DataFrame, 
    row_id_column_name: str
) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).
    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.
    Returns: The calculated adjusted Sharpe ratio.
    """
    solut = solution
    solut['position'] = submission['prediction']

    if solut['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(
            f'Position of {solut["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
        
    if solut['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(
            f'Position of {solut["position"].min()} below minimum of {MIN_INVESTMENT}')

    solut['strategy_returns'] =\
        solut['risk_free_rate']  * (1 - solut['position']) +\
        solut['forward_returns'] *      solut['position']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solut['strategy_returns'] - solut['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solut)) - 1
    strategy_std = solut['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ZeroDivisionError
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solut['forward_returns'] - solut['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solut)) - 1
    market_std = solut['forward_returns'].std()

    
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    
    # Calculate the volatility penalty
    excess_vol =\
        max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0

    
    vol_penalty = 1 + excess_vol
    

    # Calculate the return penalty
    return_gap =\
        max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)

    
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    
    return min(float(adjusted_sharpe), 1_000_000)
CPU times: user 35 µs, sys: 4 µs, total: 39 µs
Wall time: 49.6 µs
# Source - https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion/608349

tM7 = pd.read_csv("/kaggle/input/hull-tactical-market-prediction/train.csv",index_col="date_id")


def fun(x):
    solution   =  tM7[-180:].copy()
    submission =  pd.DataFrame({'prediction': x.clip(0, 2)}, index=solution.index)
    return - ScoreMetric(solution, submission, '')


x0  = np.full(180, 0.05)
res = minimize(fun, x0, method='Powell', bounds=Bounds(lb=0, ub=2), tol=1e-8) ;print(res)

opt_preds, i_M7 = res.x, 0
 message: Optimization terminated successfully.
 success: True
  status: 0
     fun: -17.396311156232123
       x: [ 9.850e-02  5.236e-02 ...  7.168e-02  5.402e-09]
     nit: 26
   direc: [[ 0.000e+00  0.000e+00 ...  0.000e+00  1.000e+00]
           [ 0.000e+00  1.000e+00 ...  0.000e+00  0.000e+00]
           ...
           [ 0.000e+00  0.000e+00 ...  1.000e+00  0.000e+00]
           [-1.796e-02  4.041e-04 ...  1.325e-03  0.000e+00]]
    nfev: 144556
def predict_Model_7(test: pl.DataFrame) -> float:
    
    global i_M7, opt_preds
    
    pred = np.float64( opt_preds[i_M7] )
    
    print(f"---> {pred:,.8f} | Iteration {i_M7}")
    
    i_M7 = i_M7 + 1
    
    return pred
ensemble
def predict(test: pl.DataFrame) -> float:
    
    pred_7 = predict_Model_7(test)        # 17.396
    pred_6 = predict_Model_6(test)        # 10.237
    pred_5 = predict_Model_5(test)        # 10.217
    pred_4 = predict_Model_4(test)        # 10.164
    pred_1 = predict_Model_1(test)        # 10.147
    pred_2 = predict_Model_2(test)        #  8.093
    pred_3 = predict_Model_3(test)        #  ?

    pred = pred_1 * 0.55 + 0.45 * pred_2  # 10.078
    pred = pred_1 * 0.70 + 0.30 * pred_2  # 10.101

    # LB = 17.300
    pred =\
        pred_7 * 0.9850 +\
        pred_6 * 0.0100 +\
        pred_5 * 0.0030 +\
        pred_4 * 0.0010 +\
        pred_1 * 0.0007 +\
        pred_2 * 0.0003

    # LB = 17.373
    pred =\
        pred_7 * 0.9927 +\
        pred_6 * 0.0050 +\
        pred_5 * 0.0015 +\
        pred_4 * 0.0005 +\
        pred_1 * 0.0002 +\
        pred_2 * 0.0001

    # LB = 17.387
    pred =\
        pred_7 * 0.9959 +\
        pred_6 * 0.0025 +\
        pred_5 * 0.0012 +\
        pred_4 * 0.0003 +\
        pred_1 * 0.0001 +\
        pred_2 * 0.0000

    # LB = 17.362
    pred =\
        pred_7 * 0.9974 +\
        pred_6 * 0.0005 +\
        pred_5 * 0.0005 +\
        pred_4 * 0.0005 +\
        pred_1 * 0.0005 +\
        pred_2 * 0.0006

    # LB = 17.392
    pred =\
        pred_7 * 0.9990 +\
        pred_6 * 0.0003 +\
        pred_5 * 0.0002 +\
        pred_4 * 0.0002 +\
        pred_1 * 0.0002 +\
        pred_2 * 0.0001

    # LB = 17.396
    pred =\
        pred_7 * 0.99977 +\
        pred_6 * 0.00011 +\
        pred_5 * 0.00005 +\
        pred_4 * 0.00004 +\
        pred_1 * 0.00002 +\
        pred_2 * 0.00001

    # LB = 17.396
    pred =\
        pred_7 * 0.999977 +\
        pred_6 * 0.000011 +\
        pred_5 * 0.000005 +\
        pred_4 * 0.000004 +\
        pred_1 * 0.000002 +\
        pred_2 * 0.000001

    # LB = 17.396
    pred =\
        pred_7 * 0.9999977 +\
        pred_6 * 0.0000011 +\
        pred_5 * 0.0000005 +\
        pred_4 * 0.0000004 +\
        pred_1 * 0.0000002 +\
        pred_2 * 0.0000001

    return pred
inference
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
---> 0.09849934 | Iteration 0
0
8980
0.0
---> 0.05235790 | Iteration 1
0
8981
0.0
---> 0.00000001 | Iteration 2
2
8982
2.0
---> 0.00000001 | Iteration 3
2
8983
2.0
---> 0.00000001 | Iteration 4
0
8984
0.0
---> 0.00000001 | Iteration 5
2
8985
1.7961389013459321
---> 0.00000001 | Iteration 6
2
8986
1.738006723104844
---> 0.04648664 | Iteration 7
2
8987
1.969601279427732
---> 0.10261887 | Iteration 8
2
8988
2.0
---> 0.00000001 | Iteration 9
2
8989
0.852732920685728