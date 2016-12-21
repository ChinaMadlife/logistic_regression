from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def file2matrix(path,delimiter):	#�ȶ���contnet���ٽ��䰴�зֿ�������eval���list����תΪ��������
	recordlist=[]
	fp=open(path,"rb")
	content=fp.read()
	fp.close()
	rowlist=content.splitlines()	#����ת��Ϊһά��
	recordlist=[map(eval,row.split(delimiter)) for row in rowlist]	#recordlistΪlist��ʽ�ģ��ٽ���ת��Ϊmat��ʽ��
	return mat(recordlist)

def logistic(wTx):
	return 1.0/(1.0+exp(-wTx))

def drawScatterbyLabel(plt,Input):	#����ɢ��ͼ
	m,n=shape(Input)
	target=Input[:,-1]
	for i in xrange(m):
		if target[i]==0:
			plt.scatter(Input[i,0],Input[i,1],c='blue',marker='o')
		else:
			plt.scatter(Input[i,0],Input[i,1],c='red',marker='s')

def buildMat(dataSet):
	m,n=shape(dataSet)
	dataMat=zeros((m,n))
	dataMat[:,0]=1
	dataMat[:,1:]=dataSet[:,:-1]
	return dataMat

Input=file2matrix("D:\\Desktop\\MLBook\\chapter05\\testSet.txt","\t")
target=Input[:,-1]
[m,n]=shape(Input)
weightlist=[]

drawScatterbyLabel(plt,Input)

dataMat=buildMat(Input)
#�˴�����ΪX0=1���ʶ���һ�ж���1
alpha=0.001
steps=500

weightmat=mat(zeros((steps,n)))

weights=ones(n)

for j in xrange(steps):
	dataIndex=range(m)
	for i in xrange(m):
		alpha=2/(1.0+i+j)+0.0001	#��̬�޸Ĳ���
		randIndex=int(random.uniform(0,len(dataIndex)))
		gradient=sum(dataMat[randIndex]*weights.T)	#ÿ��ֻȡһ�����ݽ��м���
		output=logistic(gradient)
		errors=target[randIndex]-output
		weights=weights+alpha*errors*dataMat[randIndex]
		del(dataIndex[randIndex])
	weightlist.append(weights)

print weights

k=0
for weight in weightlist:
	weightmat[k,:]=weight
	k+=1

X=linspace(0,steps,steps)

fig=plt.figure()
axes1=plt.subplot(211)
axes2=plt.subplot(212)

axes1.plot(X,-weightmat[:,0]/weightmat[:,2],color='blue',linewidth=1,linestyle="-")
axes2.plot(X,-weightmat[:,1]/weightmat[:,2],color='blue',linewidth=1,linestyle="-")
plt.show()

#we repeatedly run through the training set, and each time we encounter a training example, we update the parameters according to the gradient of the error with respect to that single training example only. This algorithm is called stochastic gradient descent (also incremental gradient descent). Whereas batch gradient descent has to scan through the entire training set before taking a single step��a costly operation if m is large��stochastic gradient descent can start making progress right away, and continues to make progress with each example it looks at