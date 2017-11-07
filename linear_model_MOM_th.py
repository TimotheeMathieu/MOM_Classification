import numpy as np
from sklearn.base import BaseEstimator,clone
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
import time
import inspect


class progressbar():
    '''Just a simple progress bar.
    '''
    def __init__(self,N):
        self.N=N
    def update(self,i):
        percent=int((i+1)/self.N*100)
        if i != self.N-1:
            print('\r'+"["+"-"*percent+' '*(100-percent)+']', end='')
        else:
            print('\r'+"["+"-"*percent+' '*(100-percent)+']')

def blockMOM(K,x):
    '''Sample the indices of K blocks for data x using a random permutation

    Parameters
    ----------

    K : int
        number of blocks

    x : array like, length = n_sample
        sample whose size correspong to the size of the sample we want to do blocks for.

    Returns 
    -------

    list of size K containing the lists of the indices of the blocks, the size of the lists are contained in [n_sample/K,2n_sample/K]
    '''
    b=int(np.floor(len(x)/K))
    nb=K-(len(x)-b*K)
    nbpu=len(x)-b*K
    perm=np.random.permutation(len(x))
    blocks=[[(b+1)*g+f for f in range(b+1) ] for g in range(nbpu)]
    blocks+=[[nbpu*(b+1)+b*g+f for f in range(b)] for g in range(nb)]
    return [perm[b] for  b in blocks]

def MOM(x,blocks):
    '''Compute the median of means of x using the blocks blocks

    Parameters
    ----------

    x : array like, length = n_sample
        sample from which we want an estimator of the mean

    blocks : list of list, provided by the function blockMOM.

    Return
    ------

    The median of means of x using the block blocks, a float.
    '''
    means_blocks=[np.mean([ x[f] for f in ind]) for ind in blocks]
    indice=np.argsort(means_blocks)[int(np.ceil(len(means_blocks)/2))]
    return means_blocks[indice],indice
class logregMOM_binary_threshold(BaseEstimator):
    '''Class of the binary classification for the logistic regression MOM.
    '''
    def __init__(self,w0=None,K=10,eta0=1,beta=1,agg=3,compter=False,progress=False,verbose=True,power=2/3,threshold=0.01,maxiter=1000,stop_delay=5):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
    def fit1(self,x,Y):
        w=np.array(self.w0)
        X=np.hstack([np.array(x),np.ones(len(x)).reshape(len(x),1)])

        pas=lambda i : 1/(1+self.eta0*i)**self.power
        if self.compter:
            self.counts=np.zeros(len(X))
        compteur=1
        fincompteur=1
        if self.progress:
            Bar=progressbar(self.epoch)
        risques=[]
        while len(risques)<self.stop_delay or np.std(risques[-self.stop_delay:])>self.threshold and compteur < self.maxiter:
            f=compteur
            if self.progress:
                Bar.update(f)
            losses=self.perte(X,Y,w)
            blocks=blockMOM(self.K,X)

            compteur+=1
            risque,b=MOM(losses,blocks)
            
            risques+=[risque]

            Xb=X[blocks[b]]
            yb=Y[blocks[b]]
            #IRLS avec regularisation L2
            eta=self.sigmoid(Xb.dot(w.reshape([len(w),1]))).reshape(len(Xb))
            D=np.diag(eta*(1-eta))
            w=w*(1-pas(f))+pas(f)*np.linalg.inv(np.transpose(Xb).dot(D).dot(Xb)+self.beta*np.eye(len(X[0]))).dot(np.transpose(Xb).dot(yb-eta)-self.beta*w)
            if self.compter:
                self.counts[blocks[b]]+=1

        print('Training finished in ',compteur,' iterations')
            
        return w

    def fit(self,x,Y):
        if self.w0 is None:
            self.w0=np.zeros(len(x[0])+1)
        y=np.array(Y).copy()
        self.values=np.sort(list(set(Y)))
        yj=y.copy()
        indmu=yj!=self.values[1]
        indu=yj==self.values[1]
        yj[indmu]=0
        yj[indu]=1
        w=np.zeros(len(self.w0))
        for f in range(self.agg):
            if self.agg !=1 and self.verbose:
                print('Passage '+str(f))
            w+=self.fit1(x,yj)
        self.w=w/self.agg

    def perte(self,X,y,w):
        pred=X.dot(w.reshape([len(w),1]))
        pred=pred.reshape(len(X))
        return np.log(1+np.exp(-(2*y-1)*pred))

    def predict(self,x):
        X=x.copy
        X=np.hstack([x,np.ones(len(x)).reshape(len(x),1)])

        pred=(X.dot(self.w.reshape([len(self.w),1]))).reshape(len(X))
        return np.array([self.values[int(p>0)] for p in pred])

    def predict_proba(self,x):
        X=x.copy
        X=np.hstack([x,np.ones(len(x)).reshape(len(x),1)])
        pred=self.sigmoid(X.dot(self.w.reshape([len(self.w)])))
        return np.array([[1-p,p] for p in pred])

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def score(self,x,y):
        pred=self.predict(x)
        return np.mean(pred==np.array(y))

