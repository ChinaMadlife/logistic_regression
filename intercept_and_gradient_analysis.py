from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def file2matrix(path,delimiter):	#先读入contnet，再将其按行分开，再用eval变成list，再转为矩阵类型
	recordlist=[]
	fp=open(path,"rb")
	content=fp.read()
	fp.close()
	rowlist=content.splitlines()	#按行转换为一维表
	recordlist=[map(eval,row.split(delimiter)) for row in rowlist]	#recordlist为list格式的，再将其转化为mat格式的
	return mat(recordlist)

def logistic(wTx):
	return 1.0/(1.0+exp(-wTx))

def drawScatterbyLabel(plt,Input):	#画出散点图
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
#此处是因为X0=1，故而第一列都是1
alpha=0.001
steps=500

weights=ones((n,1))

for k in xrange(steps):
	gradient=dataMat*mat(weights)	#此处为矢量编程的方式，gradient为m行1列的矩阵，output也是如此，只是将gradient中的每个元素都用logistic求了一次
	output=logistic(gradient)
	errors=target-output
	weights=weights+alpha*dataMat.T*errors	#一次将三个参数都更新，效率会比较低
	weightlist.append(weights)

#This method looks at every example in the entire training set on every step, and is called batch gradient descent 

print weights


weightmat=mat(zeros((steps,n)))
i=0

for weight in weightlist:	#weightlist 还是列表类型变量，只不过列表内的元素为矩阵，要将其变成矩阵类型的
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
