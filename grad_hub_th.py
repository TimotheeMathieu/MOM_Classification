import numpy as np
from sklearn.base import BaseEstimator,clone
import inspect
import statsmodels.api as sm
import statsmodels.robust.norms as norms
import statsmodels.robust.scale as scale


def huber(x):
    res= sm.RLM(x,np.ones(len(x)),M=norms.HuberT()).fit(scale_est=scale.HuberScale())
    return res.params[0]

class grad_hub():
    def __init__( self,w0=None,eta0=1e-3,beta=1e-3,threshold=0.01,c=1.35,maxiter=1000,stop_delay=5):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def psic(self,x):
        result=[ xx if np.abs(xx)<self.c else self.c*(2*(xx>0)-1) for xx in x]
        return np.array(result)

    def dpsic(self,x):
        result=[ 1 if np.abs(xx)<self.c else 0 for xx in x]
        return result
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def dmu(self,w,x,y):
        pertes=np.log(1+np.exp(-x.dot(w)*y))
        mut=huber(pertes)
        psip=self.dpsic(pertes)
        return np.array([np.sum([x[i][j]*self.sigmoid(x[i].dot(w)*y[i])*psip[i] for i in range(len(x))])/np.sum(psip) for j in range(len(x[0]))])+self.beta*w
        
    def fit(self,X,y):
        if self.w0 is None :
            self.w0=np.zeros(len(X[0]))
        w=self.w0
        pas = lambda t : 1/(1+self.eta0*t)
        risques=[]
        compteur=0
        while len(risques)<self.stop_delay or np.std(risques[-self.stop_delay:])>self.threshold and compteur < self.maxiter:
            t=compteur
            w=w+pas(t)*self.dmu(w,X,y)
            compteur+=1
            self.w=w
            risques+=[self.perte(X,y)]

            #if (t%10)==0:
            #    print('epoch ',t,' risque de ',huber(np.log(1+np.exp(-X.dot(w)*y))))
        self.w=w
        print('Training finished in ',compteur,' iterations')
    def perte(self,X,y):
        return np.mean(self.predict(X)==y)
    def predict_proba(self,xtest):
        return np.array([self.sigmoid(xtest[i].dot(self.w)) for i in range(len(xtest))])
    def predict(self,xtest):
        return self.predict_proba(xtest)>1/2
