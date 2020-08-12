import yfinance as yf
import numpy as np
from hmmlearn.hmm import GaussianHMM

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as md

class SymbolDataError(Exception): pass

class Symbol(object):
    
    def __init__(self,symbol):
        self.symbol = yf.Ticker(symbol)
        self.is_fitted = False
        self.data = None
        self.model = None
        self.p_history = None
    
    def dec_obj(self):
        self.is_fitted = False
        self.data = None
        self.model = None
        self.p_history = None
    
    def volume(self):
        return self.get_indtype(self.data.Volume.values.reshape(-1,1),"volume")
            
    def open(self):
        return self.get_indtype(self.data.Open.values.reshape(-1,1),"opening")
    
    def close(self):
        return self.get_indtype(self.data.Close.values.reshape(-1,1),"closing")
        
    def high(self):
        return self.get_indtype(self.data.High.values.reshape(-1,1),"high")
    
    def low(self):
        return self.get_indtype(self.data.Low.values.reshape(-1,1),"low")
    
    def dates(self):
        return self.get_indtype(self.data.Datetime.values.reshape(-1,1),"dates")
    
    def get_indtype(self,data,err):
        err_template = lambda error: "Use get_indicators method to init data before retrieving {0}...".format(error)
        if self.data is not None:
            return data
        else:
            raise SymbolDataError(err_template(err))
    
    def get_indicators(self,return_data = False,**kwargs):
        self.dec_obj()
        self.data = self.symbol.history(**kwargs)
        self.data['Datetime'] = self.data.index.values
        if return_data:
            return self.data
    
    def anomaly(self,
                indtype,
                factor=1.,
                return_anomalies=False,
                new_history=False,
                label_anomalous=True,
                plot=True,**kwargs):
        #naive anomalous data detection 
        if new_history:
            self.get_indicators(**kwargs)
        else:
            anomalous = (indtype > factor*np.std(indtype))
            labels = [self.dates()[k] for k in range(len(self.dates())) if anomalous[k]]
            vol = [self.volume()[k] for k in range(len(self.dates())) if anomalous[k]]
            anomalous_ind = indtype * anomalous
            if plot:
                plt.figure(figsize=(20,10))
                plt.title(self.symbol.ticker)
                plt.plot_date(
                    self.dates(),
                    anomalous_ind,
                    fmt="."
                )
                plt.axhline(y=factor*np.std(indtype),color='r')
                if label_anomalous:
                    for k in range(len(labels)):
                        plt.text(labels[k],vol[k],labels[k][0],fontsize=14)
        if return_anomalies:
            anomalous_ind = np.nonzero(anomalous_ind)
            anomalous_ind = indtype[anomalous_ind[0]]
            return (labels,anomalous_ind)
    
    def anomaly_distribution(self,indtype,factor=1.):
        _,anomalies = self.anomaly(indtype,return_anomalies=True,plot=False)
        #print(anomalies)
        plt.figure(figsize=(20,10))
        plt.title(self.symbol.ticker)
        plt.hist(anomalies,density=False)
        
    def fit(self,X,n_components=4,covariance_type="diag",n_iters=1000):
        print("Fitting model....")
        self.model = GaussianHMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        n_iter=n_iters
                        ).fit(X)
        self.is_fitted = True
        print("Fitted model....")
        print(self.model)
    
    def predict(self,X):
        self.p_history = self.model.predict(X)
        print(self.p_history)
        return self.p_history
    
    def plot_hidden_states(self,indtype):
        dates = self.dates()
        hidden_states = self.p_history
        
        if self.is_fitted:
            fig,axx = plt.subplots(self.model.n_components,figsize=(20,10))
            for i in range(self.model.n_components):
                mask = hidden_states == i
                axx[i].plot_date(dates[mask],indtype[mask])
                axx[i].set_ylim([0,max(indtype)])
                axx[i].set_xlim([dates[0],dates[len(dates)-1]])
                axx[i].set_title("Hidden State {0}".format(i))
            plt.show()
        else:
            print("A model isn't fitted, you need to fit one to plot the hidden states.")
