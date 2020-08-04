import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('seaborn-whitegrid')

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
from statsmodels.tsa.arima_model import ARIMA

matplotlib.rcParams['axes.labelsize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['text.color'] = 'k'


df = pd.read_csv('normalized_monthly_sales_product_type.csv')
product = df.loc[df['Category'] == 'Total'] #choose what productline to forecast or else total for all monthly sales


from datetime import datetime
con=product['Month']
product['Month']=pd.to_datetime(product['Month'])
product.set_index('Month', inplace=True)

product = product.groupby('Month')['Total'].sum().reset_index()

product = product.set_index('Month')
product.index

y = product['Total']

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive', freq=12)
fig = decomposition.plot()
plt.show()


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


warnings.filterwarnings("ignore") # specify to ignore warning messages
AIC_list = pd.DataFrame({}, columns=['param','param_seasonal','AIC'])
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            temp = pd.DataFrame([[ param ,  param_seasonal , results.aic ]], columns=['param','param_seasonal','AIC'])
            AIC_list = AIC_list.append( temp, ignore_index=True)  # DataFrame append list append
            del temp
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
         #print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))


m = np.amin(AIC_list['AIC'].values) # Find minimum value in AIC
l = AIC_list['AIC'].tolist().index(m) # Find index number for lowest AIC
Min_AIC_list = AIC_list.iloc[l,:]



mod = sm.tsa.statespace.SARIMAX(y,
                                order=Min_AIC_list['param'],
                                seasonal_order=Min_AIC_list['param_seasonal'],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print("### Min_AIC_list ### \n{}".format(Min_AIC_list))

print(results.summary())

pred = results.get_prediction(start=pd.to_datetime('2019-11-30'), dynamic=False)


pred_uc = results.get_forecast(steps=16)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Month')
ax.set_ylabel('Sales')
plt.legend()
plt.show()
print(pred_uc.predicted_mean)
