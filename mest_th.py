import numpy as np

from sklearn.base import BaseEstimator
import inspect

from sklearn.linear_model import LogisticRegression 

class logreg_mest(BaseEstimator):

    def __init__(self,beta=0,threshold=0.01,maxiter=1000,stop_delay=5):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self,X,y):
        clf=LogisticRegression()
        clf.fit(X,y)
        self.w=clf.coef_
        risques=[]
        compteur=0
        while len(risques)<self.stop_delay or np.std(risques[-self.stop_delay:])>self.threshold and compteur < self.maxiter:
            e=y-self.predict(X)
            sigma=1.4826*np.median(np.abs(e-np.median(e)))
            u=e/(sigma+1e-10)
            W=(1-(u/4.685)**2)**2*(np.abs(u)<4.685)*np.eye(len(X))
            self.w=np.linalg.inv(np.transpose(X).dot(W).dot(X)).dot(np.transpose(X).dot(W).dot(y))
            compteur+=1
            risques+=[self.perte(X,y)]
        print('Training finished in ',compteur,' iterations')
    def perte(self,X,y):
        return np.mean(self.predict(X)==y)

    def predict_proba(self,x):
        return 1/(1+np.exp(-self.w.dot(np.transpose(x))))

    def predict(self,x):
        pred=self.predict_proba(x)
        return np.floor(pred>=0.5)


