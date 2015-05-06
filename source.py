
import pandas as pd
import time
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as DV
import numpy as np
from sklearn.metrics import confusion_matrix

#members = pd.read_csv('members.tsv',sep='\t')
#emails = pd.read_csv('emails.tsv',sep='\t')#, parse_dates = [1])
#responses = pd.read_csv('email_responses.tsv',sep='\t')
##==============================================================================
## f = open('members.tsv', 'r')
## i=0 
## while i!=973032:
##      f.readline()
##      i+=1
## f.readline()
##==============================================================================
#
##==============================================================================
## Experiment to see what is the percentage of the abnormal action(eg. click without open). It is email based. It is estimated by using the responded emails. We do not include the ‘just open’ into denominator since we are not sure how many opened emails did not get log. So we use ‘click without unsub’+’unsub’ as denominator. It is about 43.8%
##==============================================================================
##==============================================================================
## def transform2(x):
##     for w in x:
##         if w=='open':
##             return 1
##         if w=='unsub' or w=='click':
##             return 0
##==============================================================================
##==============================================================================
## def transform2(x):
##     if not x.empty:
##         flag=True
##         result=1 #'open'
##         for w in x:
##             if flag:
##                 if w=='unsub' or w=='click':
##                     return 0
##                 flag=False        
##             if w=='click':
##                 result= 2
##                 continue
##             if w=='unsub':
##                 result=-1
##                 continue
##             if w=='open':
##                 continue
##             else:
##                 return None
##         return result
##               
## 
## email_res0_merged=pd.merge(emails,responses,how='left',on='email_id')
## email_res0_member_merged=pd.merge(email_res0_merged,members,how='left',on='member_id')
## e_r0_m_merged_eid_group=email_res0_member_merged.groupby('email_id')
## tempagg=e_r0_m_merged_eid_group.agg({'action':transform2})
## 
## 
## email_member_merged=pd.merge(emails,members,how='left',on='member_id')
## e_m_merged_plus_act_normal=email_member_merged
## e_m_merged_plus_act_normal['is_normal_action']=tempagg['action'].values
## 
## len(e_m_merged_plus_act_normal[e_m_merged_plus_act_normal['is_normal_action']==0])
## len(e_m_merged_plus_act_normal[e_m_merged_plus_act_normal['is_normal_action']==-1])
## len(e_m_merged_plus_act_normal[e_m_merged_plus_act_normal['is_normal_action']==2])
##==============================================================================
##float(105717)/(228229+12808)
##Out[239]: 0.4385924152723441
#
##==============================================================================
## Transform to get the target variable. Could be changed to be different metric
##==============================================================================
#def transform(x):
#    clickflag=False
#    for w in x:
#        if w=='click':
#            clickflag=True
#        if w=='unsub':
#            return -1
#    if clickflag:
#            return 1
#    return 0
#    
#def getJoinedDays(s):
#    i=s.find('tplus')
#    j=s.find('_',i)
#    if j==-1:
#        return s[i+5:]
#    return s[i+5:j]
#    
#    
#criterion = emails['variants'].map(lambda x: x.startswith('T plus N'))
#emails_TplusN=emails[criterion]
#emails_TplusN=emails_TplusN.reset_index()
#
#
#alert_or_cloud=[(1 if 'cloud' in w else 0) for w in emails_TplusN['variants']]
#days_joined=map(getJoinedDays,emails_TplusN['variants'])
#
#emails_TplusN['alert_or_cloud']=alert_or_cloud
#emails_TplusN['days_joined']=days_joined
#
#act_agg=responses.groupby('email_id').agg({'action':transform})
#e_r_merged=pd.merge(emails_TplusN,act_agg,how='left',left_on='email_id',right_index=True)
#e_r_merged['action'].fillna(0,inplace=True) #convert the nonresponse to be the same as pure open which is 0
#
##e_r_merged['action'].value_counts()
##Out[308]: 
## 0    8418192
## 1     276869
##-1      13557
##dtype: int64
#
#e_r_m_merged=pd.merge(e_r_merged,members,on='member_id')
#e_r_m_merged['Epoch_second']=[time.mktime(time.strptime(w, "%Y-%m-%d %H:%M:%S")) for w in e_r_m_merged['sent_time']]
#e_r_m_merged['day']=[time.strptime(w, "%Y-%m-%d %H:%M:%S").tm_mday for w in e_r_m_merged['sent_time']]
#e_r_m_merged['weekday']=[time.strptime(w, "%Y-%m-%d %H:%M:%S").tm_wday for w in e_r_m_merged['sent_time']]
#e_r_m_merged['hour']=[time.strptime(w, "%Y-%m-%d %H:%M:%S").tm_hour for w in e_r_m_merged['sent_time']]
#e_r_m_merged['minute']=[time.strptime(w, "%Y-%m-%d %H:%M:%S").tm_min for w in e_r_m_merged['sent_time']]
#e_r_m_merged['graduate_age'] = 2013 - e_r_m_merged['hs_or_ged_year']   
#e_r_m_merged.to_csv('e_r_m_merged_full_feature.csv',index=False)
#
#
#
#Xylabel=['alert_or_cloud', 'days_joined', 'email_domain',
#        'zip', 'degree_level',
#       'pcp_score', 'Epoch_second', 'day',
#       'weekday', 'hour', 'minute', 'graduate_age','action','keyword']
#
#Xy=e_r_m_merged[Xylabel]       
#Xy[pd.isnull(Xy['email_domain'])|(Xy['email_domain']=='\N')]
#Xy[pd.isnull(Xy['days_joined'])|(Xy['days_joined']=='')]
#Xy=Xy[pd.notnull(Xy['zip'])&(Xy['zip']!='\N')]
#Xy.to_csv('Xy.csv',index=False)
##len(train_X[pd.isnull(train_X['degree_level'])|(train_X['degree_level']=='\N')])
##Out[395]: 149692
##train_X.shape
##Out[396]: (8360562, 12)
#
##==============================================================================
## preprocessing the data
##==============================================================================
#le1 = preprocessing.LabelEncoder()
#Xy['email_domain_num']=le1.fit_transform(Xy['email_domain'])
#
#le3 = preprocessing.LabelEncoder()
#Xy['keyword_num']=le3.fit_transform(Xy['keyword'])
#
#le2 = preprocessing.LabelEncoder()
#Xy['degree_level_num']=le2.fit_transform(Xy['degree_level'])
#enc = preprocessing.OneHotEncoder(sparse=False)
#degree_level_vectors=enc.fit_transform(Xy['degree_level_num'].values.reshape((len(Xy['degree_level_num'].values),1)))
#
#Xlabel=['alert_or_cloud', 'days_joined','Epoch_second', 'day','weekday', 'hour', 'minute','email_domain_num','graduate_age','keyword_num']
#       # 'zip' need to delete something like 'K2B6A1'
#
#
#X=np.append(Xy[Xlabel].values,degree_level_vectors,axis=1)
#y=Xy['action'].values
##from types import *
##for i in range(0,X.shape[0]):
##    for j in range(0,X.shape[1]):
##        if (type(X[i][j]) is not IntType) and (type(X[i][j]) is not FloatType):
##            if not X[i][j].isdigit():
##                print (i,j)
##the following index has bad days_joined
#delete_index=[761949,808740,1140993,1214501,1291062,1509294,1755951,2053344,2192183,2511532,3212968,3709596,3711119,3889168,3912266,3965875,4260620,4389753,4513041,4734428,4758864,4781965,4803037,4995586,5005480,5129268,5380662,5468721,5489244,5610481,7006903,7052171]
#X=np.delete(X,delete_index,0)
#y=np.delete(y,delete_index,0)
#X=X.astype(int)
#y=y.astype(int)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y)
#
##==============================================================================
## balance data
##==============================================================================
#clickindex=[]
#passiveindex=[]
#unsubindex=[]
#for i in range(0,len(y_train)):
#    if y_train[i]==1:
#        clickindex.append(i)
#    if y_train[i]==0:
#        passiveindex.append(i)
#    if y_train[i]==-1:
#        unsubindex.append(i)
#clickindex=(np.random.choice(clickindex, len(unsubindex))).tolist()
#passiveindex=(np.random.choice(passiveindex, len(unsubindex))).tolist()
#
#X_train_b=X_train[clickindex+passiveindex+unsubindex]
#y_train_b=y_train[clickindex+passiveindex+unsubindex]
#
#
#clickindex=[]
#passiveindex=[]
#unsubindex=[]
#for i in range(0,len(y_test)):
#    if y_test[i]==1:
#        clickindex.append(i)
#    if y_test[i]==0:
#        passiveindex.append(i)
#    if y_test[i]==-1:
#        unsubindex.append(i)
#clickindex=(np.random.choice(clickindex, len(unsubindex))).tolist()
#passiveindex=(np.random.choice(passiveindex, len(unsubindex))).tolist()
#
#X_test_b=X_test[clickindex+passiveindex+unsubindex]
#y_test_b=y_test[clickindex+passiveindex+unsubindex]
#
##==============================================================================
## train model
##==============================================================================
##from sklearn.neighbors import KNeighborsClassifier
##knn = KNeighborsClassifier()
##knn.fit(X_train, y_train)
#
##==============================================================================
## normal or standard
#
#
#
##==============================================================================
#
#from sklearn.linear_model import LogisticRegression
#logi=LogisticRegression()
#logi.fit(X_train_b, y_train_b)
#logi.score(X_test,y_test)
#pred_logi=logi.predict(X_test)
#confusion_matrix(y_test, pred_logi)
#
#from sklearn import tree
#dtree = tree.DecisionTreeClassifier()   #0.44011505487928282
#dtree.fit(X_train_b, y_train_b)
#dtree.score(X_test,y_test)
#pred_dtree=dtree.predict(X_test)
#confusion_matrix(y_test, pred_dtree)
##from sklearn.externals.six import StringIO  
##import pydot 
##dot_data = StringIO() 
##tree.export_graphviz(dtree, out_file=dot_data) 
##graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
##graph.write_pdf("dtree.pdf") 
#
#
#
#with open("dtree.dot", 'w') as f:
#    f = tree.export_graphviz(dtree, out_file=f)
#    
#from sklearn.ensemble import RandomForestClassifier #0.50312635607399148
#rf=RandomForestClassifier()
#rf.fit(X_train_b, y_train_b)
#rf.score(X_test,y_test)
#pred_rf=rf.predict(X_test)
#confusion_matrix(y_test, pred_rf)
#
#from sklearn.neighbors import KNeighborsClassifier #0.43318726607349867
#knn=KNeighborsClassifier()
#knn.fit(X_train_b, y_train_b)
#knn.score(X_test,y_test)
#pred_knn=knn.predict(X_test)
#confusion_matrix(y_test, pred_knn)
#
#from sklearn.naive_bayes import GaussianNB #0.39196883643289687
#gnb=GaussianNB()
#gnb.fit(X_train_b, y_train_b)
#gnb.score(X_test,y_test)
#pred_gnb=knn.predict(X_test)
#confusion_matrix(y_test, pred_gnb)
#
#from sklearn.naive_bayes import MultinomialNB #0.56146139982479581
#mnb=MultinomialNB()
#mnb.fit(X_train_b, y_train_b)
#mnb.score(X_test,y_test)
#pred_mnb=mnb.predict(X_test)
#confusion_matrix(y_test, pred_mnb)
#
#from sklearn.naive_bayes import BernoulliNB  #0.51000008133453711
#bnb=BernoulliNB()
#bnb.fit(X_train_b, y_train_b)
#bnb.score(X_test,y_test)
#pred_bnb=bnb.predict(X_test)
#confusion_matrix(y_test, pred_bnb)
#
#from sklearn.lda import LDA #0.53491715598959488 UserWarning: Variables are collinear.
#lda=LDA()
#lda.fit(X_train_b, y_train_b)
#lda.score(X_test,y_test)
#pred_lda=lda.predict(X_test)
#confusion_matrix(y_test, pred_lda)
#
#from sklearn import svm
#svc = svm.SVC(kernel='linear')
#svc.fit(X_train, y_train)
#svc.score(X_test,y_test)
#
#
#
#pred_logi=svc.predict(X_test)
#confusion_matrix(y_test, pred_logi)
#
##for clf, name in (
##        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
##        (Perceptron(n_iter=50), "Perceptron"),
##        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
##        (KNeighborsClassifier(n_neighbors=10), "kNN"),
##        (RandomForestClassifier(n_estimators=100), "Random forest")):
##
#
#from sklearn import linear_model
#sgd = linear_model.SGDClassifier(class_weight={-1:10,0:1,1:5},penalty='elasticnet')
#sgd.fit(X_train_b, y_train_b)
#sgd.score(X_test,y_test)
#pred_sgd=sgd.predict(X_test)
#confusion_matrix(y_test, pred_sgd)
#
#
#from sklearn.ensemble import RandomForestClassifier #0.50312635607399148
#rfw=RandomForestClassifier(class_weight='auto')#{-1:100,0:1,1:5})
#rfw.fit(X_train_b, y_train_b)
#rfw.score(X_test,y_test)
#pred_rfw=rfw.predict(X_test)
#confusion_matrix(y_test, pred_rfw)

from sklearn import tree
dtreew = tree.DecisionTreeClassifier(class_weight={-1:0.5,0:10000,1:5})   #0.44011505487928282
dtreew.fit(X_train_b, y_train_b)
dtreew.score(X_test,y_test)
pred_dtreew=dtreew.predict(X_test)
confusion_matrix(y_test, pred_dtreew)
