# Hull Tactical - Market Prediction

Can you predict market predictability?

Overview
Your task is to predict the stock market returns as represented by the excess returns of the S&P 500 while also managing volatility constraints. Your work will test the Efficient Market Hypothesis and challenge common tenets of personal finance.

Start

2 months ago
Close
a month to go
Merger & Entry
Description
Wisdom from most personal finance experts would suggest that it's irresponsible to try and time the market. The Efficient Market Hypothesis (EMH) would agree: everything knowable is already priced in, so don’t bother trying.

But in the age of machine learning, is it irresponsible to not try and time the market? Is the EMH an extreme oversimplification at best and possibly just…false?

This competition is about more than predictive modeling. Predicting market returns challenges the assumptions of market efficiency. Your work could help reshape how investors and academics understand financial markets. Participants could uncover signals others overlook, develop innovative strategies, and contribute to a deeper understanding of market behavior—potentially rewriting a fundamental principle of modern finance. Most investors don’t beat the S&P 500. That failure has been used for decades to prop up EMH: If even the professionals can’t win, it must be impossible. This observation has long been cited as evidence for the Efficient Market Hypothesis the idea that prices already reflect all available information and no persistent edge is possible. This story is tidy, but reality is less so. Markets are noisy, messy, and full of behavioral quirks that don’t vanish just because academic orthodoxy said they should.

Data science has changed the game. With enough features, machine learning, and creativity, it’s possible to uncover repeatable edges that theory says shouldn’t exist. The real challenge isn’t whether they exist—it’s whether you can find them and combine them in a way that is robust enough to overcome frictions and implementation issues.

Our current approach blends a handful of quantitative models to adjust market exposure at the close of each trading day. It points in the right direction, but with a blurry compass. Our model is clearly a sub-optimal way to model a complex, non-linear, adaptive system. This competition asks you to do better: to build a model that predicts excess returns and includes a betting strategy designed to outperform the S&P 500 while staying within a 120% volatility constraint. We’ll provide daily data that combines public market information with our proprietary dataset, giving you the raw material to uncover patterns most miss.

Unlike many Kaggle challenges, this isn’t just a theoretical exercise. The models you build here could be valuable in live investment strategies. And if you succeed, you’ll be doing more than improving a prediction engine—you’ll be helping to demonstrate that financial markets are not fully efficient, challenging one of the cornerstones of modern finance, and paving the way for better, more accessible tools for investors.

Evaluation
The competition's metric is a variant of the Sharpe ratio that penalizes strategies that take on significantly more volatility than the underlying market or fail to outperform the market's return. The metric code is available here.

Hull Competition Sharpe:
```python
import numpy as np
import pandas as pd
import pandas.api.types

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).

    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.

    Returns:
        float: The calculated adjusted Sharpe ratio.
    """

    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    solution = solution
    solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}')

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)
```


Submission File
You must submit to this competition using the provided evaluation API, which ensures that models do not peek forward in time. For each trading day, you must predict an optimal allocation of funds to holding the S&P500. As some leverage is allowed, the valid range covers 0 to 2. See this example notebook for more details.

Demo Example Notebook:

```python
"""
The evaluation API requires that you set up a server which will respond to inference requests.
We have already defined the server; you just need write the predict function.
When we evaluate your submission on the hidden test set the client defined in `default_gateway` will run in a different container
with direct access to the hidden test set and hand off the data timestep by timestep.

Your code will always have access to the published copies of the copmetition files.
"""

import os

import pandas as pd
import polars as pl

import kaggle_evaluation.default_inference_server


def predict(test: pl.DataFrame) -> float:
    """Replace this function with your inference code.
    You can return either a Pandas or Polars dataframe, though Polars is recommended for performance.
    Each batch of predictions (except the very first) must be returned within 5 minutes of the batch features being provided.
    """
    return 0.0


# When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting
# or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very
# first `predict` call, which does not have the usual 1 minute response deadline.
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/hull-tactical-market-prediction/',))
```

Dataset Description
This competition challenges you to predict the daily returns of the S&P 500 index using a tailored set of market data features.

Competition Phases and Data Updates
The competition will proceed in two phases:

A model training phase with a test set of six months of historical data. Because these prices are publicly available leaderboard scores during this phase are not meaningful.
A forecasting phase with a test set to be collected after submissions close. You should expect the scored portion of the test set to be about the same size as the scored portion of the test set in the first phase.
During the forecasting phase the evaluation API will serve test data from the beginning of the public set to the end of the private set. This includes trading days before the submission deadline, which will not be scored. The first date_id served by the API will remain constant throughout the competition.

Files
train.csv Historic market data. The coverage stretches back decades; expect to see extensive missing values early on.

date_id - An identifier for a single trading day.
M* - Market Dynamics/Technical features.
E* - Macro Economic features.
I* - Interest Rate features.
P* - Price/Valuation features.
V* - Volatility features.
S* - Sentiment features.
MOM* - Momentum features.
D* - Dummy/Binary features.
forward_returns - The returns from buying the S&P 500 and selling it a day later. Train set only.
risk_free_rate - The federal funds rate. Train set only.
market_forward_excess_returns - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4. Train set only.
test.csv A mock test set representing the structure of the unseen test set. The test set used for the public leaderboard set is a copy of the last 180 date IDs in the train set. As a result, the public leaderboard scores are not meaningful. The unseen copy of this file served by the evaluation API may be updated during the model training phase.

date_id
[feature_name] - The feature columns are the same as in train.csv.
is_scored - Whether this row is included in the evaluation metric calculation. During the model training phase this will be true for the first 180 rows only. Test set only.
lagged_forward_returns - The returns from buying the S&P 500 and selling it a day later, provided with a lag of one day.
lagged_risk_free_rate - The federal funds rate, provided with a lag of one day.
lagged_market_forward_excess_returns - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4, provided with a lag of one day.
kaggle_evaluation/ Files used by the evaluation API. See the demo submission for an illustration of how to use the API.

Once the competition ends, we will periodically publish our data on our website, and you're welcome to use it for your own trading

Data Samples

train.csv:
date_id,D1,D2,D3,D4,D5,D6,D7,D8,D9,E1,E10,E11,E12,E13,E14,E15,E16,E17,E18,E19,E2,E20,E3,E4,E5,E6,E7,E8,E9,I1,I2,I3,I4,I5,I6,I7,I8,I9,M1,M10,M11,M12,M13,M14,M15,M16,M17,M18,M2,M3,M4,M5,M6,M7,M8,M9,P1,P10,P11,P12,P13,P2,P3,P4,P5,P6,P7,P8,P9,S1,S10,S11,S12,S2,S3,S4,S5,S6,S7,S8,S9,V1,V10,V11,V12,V13,V2,V3,V4,V5,V6,V7,V8,V9,forward_returns,risk_free_rate,market_forward_excess_returns
0,0,0,0,1,1,0,0,0,1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.0024211742695206,0.000300793650793651,-0.00303847935997865
1,0,0,0,1,1,0,0,0,1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.00849467703679185,0.000302777777777778,-0.00911404561931339
2,0,0,0,1,0,0,0,0,1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.00962447844228109,0.000301190476190476,-0.0102425375009931
3,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,0.004662397483429,0.000299206349206349,0.00404620350408207
4,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.0116857701984905,0.000299206349206349,-0.0123006546540279
5,0,0,0,1,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,-0.00644942294636774,0.000299603174603175,-0.00706593438603214

test.csv:
date_id,D1,D2,D3,D4,D5,D6,D7,D8,D9,E1,E10,E11,E12,E13,E14,E15,E16,E17,E18,E19,E2,E20,E3,E4,E5,E6,E7,E8,E9,I1,I2,I3,I4,I5,I6,I7,I8,I9,M1,M10,M11,M12,M13,M14,M15,M16,M17,M18,M2,M3,M4,M5,M6,M7,M8,M9,P1,P10,P11,P12,P13,P2,P3,P4,P5,P6,P7,P8,P9,S1,S10,S11,S12,S2,S3,S4,S5,S6,S7,S8,S9,V1,V10,V11,V12,V13,V2,V3,V4,V5,V6,V7,V8,V9,is_scored,lagged_forward_returns,lagged_risk_free_rate,lagged_market_forward_excess_returns
8980,0,0,0,0,1,0,0,1,0,1.5776511229575,0.186177248677249,0.00132275132275132,0.00132275132275132,0.00132275132275132,0.00132275132275132,0.955026455026455,-0.583419489571196,-0.704263953275144,0.298364953667522,-0.691360591003745,1.25906537584674,1.55651624583326,1.71258010208486,0.0330687830687831,0.333333333333333,0.0363756613756614,-0.0464826902005396,-0.312325656730028,0.913029100529101,0.306216931216931,1.02575637127463,0.0813492063492064,0.478174603174603,0.675626892422322,0.69973544973545,0.256283068783069,0.360449735449735,0.676060977652112,-1.31043784734965,-0.392342451544124,-0.236099720197407,0.169553887632956,-1.19641906139403,-0.760030391853607,0.641203703703704,0.916666666666667,0.28968253968254,0.318783068783069,-0.321980913607935,3.39480084198328,0.110938701755659,0.755237009958536,-0.318948475755792,-0.596875321395981,0.86739417989418,0.348006347061464,0.470568783068783,2.11367466392798,2.24781993199633,0.882723840624582,0.794642857142857,-1.42783361786944,0.352513227513228,0.926256613756614,0.431382936402851,-0.476975826432751,0.500245409318266,1.78417266017572,0.0297619047619048,0.294718644225946,0.514550264550265,0.446428571428571,0.466550931948929,0.0857167792140139,-0.230132159589618,0.272486772486772,-0.10689351649087,0.19973544973545,0.409391534391534,0.532717071849416,0.744047619047619,0.44047619047619,-0.654839290601487,0.69973544973545,0.699074074074074,-0.502400056469275,0.882936507936508,0.892195767195767,0.828042328042328,0.999172490701155,0.759920634920635,-0.803127021856731,0.170965608465608,-0.751909229513165,true,0.00354143791658279,0.000161111111111111,0.0030684383816803
8981,0,0,0,0,1,0,0,1,0,1.57518198384718,0.185846560846561,0.000992063492063492,0.000992063492063492,0.000992063492063492,0.000992063492063492,0.955357142857143,-0.583073938590698,-0.703758718303382,0.297608278852814,-0.504499349198961,1.19346789571564,1.55418371129725,1.6400541928101,0.0327380952380952,0.333002645502646,0.0360449735449735,0.0735824914975662,-0.312344884449625,0.913359788359788,0.305886243386243,0.989571475584055,0.0826719576719577,0.477843915343915,0.661526867067106,0.71957671957672,0.255952380952381,0.361111111111111,0.660556174585453,-1.2346873950993,-0.357219699489025,-0.220753939458671,0.191397845454955,-1.20352553569629,-0.759527355825838,0.641865079365079,0.852513227513228,0.289021164021164,0.318452380952381,0.380119105907947,0.298001409341175,0.0100208544199729,1.12665327563392,-0.314880589648339,-0.597244605195331,0.961640211640212,0.313526799311268,0.331018518518519,2.08955191071943,2.19927828408676,-0.428140108889947,0.711309523809524,-1.37652034185826,0.953042328042328,0.386904761904762,0.523548897538259,-0.421364705007851,-0.234829322126506,1.77017480321353,0.0337301587301587,0.304496398426839,0.638227513227513,0.636904761904762,1.84910061989064,0.28169041152573,-0.0419952878066536,0.448412698412698,0.0943209637170576,0.215608465608466,0.409391534391534,0.597863671921122,0.872354497354497,0.691137566137566,-0.583442629041542,0.629960317460317,0.598544973544973,-0.394268392721835,0.863756613756614,0.699074074074074,0.831349206349206,1.12033590447645,0.556216931216931,-0.68619206710584,0.141865079365079,-0.660325922143153,true,-0.00596375516242376,0.000161587301587302,-0.00643673485605641
8982,0,0,0,0,1,0,0,0,1,1.57272019361418,0.185515873015873,0.000661375661375661,0.000661375661375661,0.000661375661375661,0.000661375661375661,0.955687830687831,-0.0833560297674705,-0.573546422064747,0.225821989405292,-0.393903252241081,1.12336135196621,1.55172281157436,1.56272237701895,0.0324074074074074,0.332671957671958,0.0357142857142857,0.0335814772164045,-0.312364112493139,0.913690476190476,0.291997354497354,1.04051432347834,0.0813492063492064,0.477513227513228,0.655741185085323,0.724206349206349,0.226190476190476,0.355820105820106,0.650325267809456,-1.12898877844879,-0.34729669520442,-0.241044438679655,0.199469140819869,-1.18959207249406,-0.757390989658372,0.652116402116402,0.87962962962963,0.288359788359788,0.318121693121693,1.35133365344964,0.391479918853056,-0.233157734903627,0.860735273975526,-0.244248397621428,-0.596983103445657,0.513227513227513,0.276338219251506,0.348544973544974,2.06740588968839,2.14958454629629,0.368687836689561,0.212632275132275,-1.34762033509144,0.210978835978836,0.635251322751323,-1.13819789214957,-0.494248419264684,-1.04271762453638,1.75417145414358,0.0324074074074074,0.257609067572841,0.082010582010582,0.152116402116402,-0.201611423592575,0.346372627193352,0.0540323719644936,0.137566137566138,0.294305483207499,0.194444444444444,0.409391534391534,0.596527956212319,0.778439153439153,0.634920634920635,-0.483235554092685,0.669973544973545,0.603835978835979,-0.17041975044923,0.848544973544973,0.647486772486772,0.832671957671958,1.08899157723919,0.665343915343915,-0.459367235493119,0.199404761904762,-0.510979013003655,true,-0.007410278273002,0.000160436507936508,-0.0078819524110791
8983,0,0,0,0,1,0,0,0,1,1.57026571131413,0.185185185185185,0.0198412698412698,0.0198412698412698,0.00661375661375661,0.00661375661375661,0.956018518518518,-0.0834026059234492,-0.573179678786274,0.225094329220265,-0.541646066635981,1.16798166054575,1.54927187468707,1.61121453177099,0.0320767195767196,0.33234126984127,0.0353835978835979,0.0339981229409789,-0.312383340860597,0.914021164021164,0.267195767195767,1.09137578058286,0.085978835978836,0.47718253968254,0.648582284927489,0.728174603174603,0.23015873015873,0.351851851851852,0.649527730028477,-1.15043068206866,-0.300916851964682,-0.145170887455168,0.269388832067044,-1.1651388929877,-0.754947782812743,0.652116402116402,0.853835978835979,0.279431216931217,0.317791005291005,-0.212994533935115,3.03272542222693,0.0241575963800274,1.42699427862205,-0.286941672837688,-0.572019322685127,0.681547619047619,0.302134301444194,0.309193121693122,2.07941458615872,2.17570531629265,0.368943230575332,0.660714285714286,-1.39941056954107,0.724867724867725,0.216269841269841,-0.230556755699477,-0.318347080219475,0.396917420819882,1.7696782090854,0.0337301587301587,0.202349617972462,0.324074074074074,0.212962962962963,0.0528780799084826,-0.0490228810760025,0.120827754819863,0.21957671957672,0.137942310970332,0.167328042328042,0.409391534391534,0.579725824997979,0.44973544973545,0.665343915343915,-0.546297914473504,0.590608465608466,0.558862433862434,-0.275099014032746,0.826058201058201,0.445767195767196,0.835978835978836,1.04098835147996,0.59457671957672,-0.561643257520821,0.161706349206349,-0.57599675443862,true,0.00541958861105485,0.00016,0.00494875574281902
8984,0,0,0,0,0,0,1,0,1,1.56781849632359,0.184854497354497,0.019510582010582,0.019510582010582,0.00628306878306878,0.00628306878306878,0.956349206349206,-0.0834491827679065,-0.572813072539872,0.224366396083157,-0.714549090953736,1.24371332730486,1.54254272848842,1.69360438914035,0.0317460317460317,0.332010582010582,0.0350529100529101,0.0295860850674559,-0.301854948830411,0.914351851851852,0.260912698412698,1.01157139280454,0.0939153439153439,0.476851851851852,0.63866680203438,0.729497354497355,0.238756613756614,0.353835978835979,0.6300223321331,-1.21128722209161,-0.269172835985608,-0.193709725764864,0.223267320891167,-1.17659963562673,-0.755178432649443,0.656746031746032,0.786375661375661,0.279100529100529,0.317460317460317,-1.05892899423604,0.350055000699561,0.319599178852967,1.21481583276401,-0.330839932121156,-0.571782039352097,0.681878306878307,0.345035327030027,0.353174603174603,2.09783109909893,2.23227588809677,0.646594862820099,0.326719576719577,-1.41628233323729,0.611441798941799,0.426256613756614,0.0464718999002435,-0.262085572139769,1.30414246106029,1.78551545155893,0.0396825396825397,0.23036426287256,0.583994708994709,0.44510582010582,1.37892667090251,-0.182024872967118,0.012609862422952,0.369047619047619,0.0110209572014676,0.130291005291005,0.409391534391534,0.572655522146939,0.489417989417989,0.600529100529101,-0.587258089768681,0.461309523809524,0.487433862433862,-0.395480129745472,0.807539682539683,0.707671957671958,0.83994708994709,0.944592835366502,0.715608465608466,-0.692649436933271,0.124669312169312,-0.654045499604081,true,0.0083574113772642,0.000159444444444444,0.00788658644553631

