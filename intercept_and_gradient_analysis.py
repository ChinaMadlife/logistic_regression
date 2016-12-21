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

#drawScatterbyLabel(plt,Input)

dataMat=buildMat(Input)
#�˴�����ΪX0=1���ʶ���һ�ж���1
alpha=0.001
steps=500

weights=ones((n,1))

for k in xrange(steps):
	gradient=dataMat*mat(weights)	#�˴�Ϊʸ����̵ķ�ʽ��gradientΪm��1�еľ���outputҲ����ˣ�ֻ�ǽ�gradient�е�ÿ��Ԫ�ض���logistic����һ��
	output=logistic(gradient)
	errors=target-output
	weights=weights+alpha*dataMat.T*errors	#һ�ν��������������£�Ч�ʻ�Ƚϵ�
	weightlist.append(weights)

#This method looks at every example in the entire training set on every step, and is called batch gradient descent 

print weights


weightmat=mat(zeros((steps,n)))
i=0

for weight in weightlist:	#weightlist �����б����ͱ�����ֻ�����б��ڵ�Ԫ��Ϊ����Ҫ�����ɾ������͵�
	weightmat[i,:]=weight.T
	i+=1
X=linspace(0,steps,steps)
fig=plt.figure(1)
axes1=plt.subplot(211)
axes2=plt.subplot(212)

axes1.plot(X[0:10],-weightmat[0:10,0]/weightmat[0:10,2],color='blue',linewidth=1,linestyle="-")
axes2.plot(X[10:],-weightmat[10:,0]/weightmat[10:,2],color='blue',linewidth=1,linestyle="-")

plt.show()

fig=plt.figure(2)
axes1=plt.subplot(211)
axes2=plt.subplot(212)
axes1.plot(X[0:10],-weightmat[0:10,1]/weightmat[0:10,2],color='red',linewidth=1,linestyle="-")
axes2.plot(X[10:],-weightmat[10:,1]/weightmat[10:,2],color='red',linewidth=1,linestyle="-")

plt.show()
