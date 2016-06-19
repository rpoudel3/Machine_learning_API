from api import app
import pandas as pd
import json
import os
from flask import jsonify
from flask import render_template
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as metrics
import sklearn.cross_validation as cv
from sklearn.ensemble import RandomForestClassifier


from flask import url_for


def get_abs_path():
    return os.path.abspath(os.path.dirname(__file__))
def binary(y):
    b=y
    for i in range(len(y)):
        if y[i]==2:
            b[i]=0
        else:
            b[i]=1
    return b

def get_data():
    f_name=os.path.join(get_abs_path(),'data','breast-cancer-wisconsin.csv')

    columns = ['code', 'clump_thickness', 'size_uniformity', 'shape_uniformity', 'adhesion', 'cell_size', 'bare_nuclei',
              'bland_chromatin', 'normal_nuclei', 'mitosis', 'class']
    df = pd.read_csv(f_name, sep =',', header = None, names = columns,na_values = '?' )

    return df.dropna()

@app.route('/')
def index():
    df=get_data()
    X=df.ix[:, (df.columns !='class') & (df.columns !='code')].as_matrix() #this gives a numpy
    y=df.ix[:,df.columns=='class'].as_matrix()
    #scale
    scaler=preprocessing.StandardScaler().fit(X)
    scaled=scaler.transform(X)
    pcomp=decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components=pcomp.transform(scaled)
    var=pcomp.explained_variance_ratio_.sum()
    #KMeans
    model=KMeans(init='k-means++',n_clusters=2)
    model.fit(components)
    #Plot
    fig=plt.figure()
    plt.scatter(components[:,0],components[:,1],c=model.labels_)
    centers=plt.plot(
        [model.cluster_centers_[0,0],model.cluster_centers_[1,0]],
        [model.cluster_centers_[1,0],model.cluster_centers_[1,1]],
        'kx', c='Green'
    )

    #increase size of center points
    plt.setp(centers,ms=11.0)
    plt.setp(centers,mew=1.8)

    #Plot axes adjustments
    axes=plt.gca()
    axes.set_xlim([-7.5,3])
    axes.set_ylim([-2,5])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clustering of PCS({:.2f}% Var.Explained'.format(
        var*100
    ))
    #Save fig

    fig_path=os.path.join(get_abs_path(),'static','tmp2','cluster.png')
    fig.savefig(fig_path)
    return render_template('index.html',fig=url_for('static',filename='tmp2/cluster.png'))

@app.route('/d3')
def d3():
    df=get_data()
    X=df.ix[:, (df.columns !='class') & (df.columns !='code')].as_matrix() #this gives a numpy
    y=df.ix[:,df.columns=='class'].as_matrix()
    #scale
    scaler=preprocessing.StandardScaler().fit(X)
    scaled=scaler.transform(X)
    pcomp=decomposition.PCA(n_components=2)
    pcomp.fit(scaled)
    components=pcomp.transform(scaled)
    var=pcomp.explained_variance_ratio_.sum()
    #KMeans
    model=KMeans(init='k-means++',n_clusters=2)
    model.fit(components)
    #Generate CSV
    cluster_data=pd.DataFrame(
        {'pc1':components[:,0],
         'pc2':components[:,1],
         'labels':model.labels_}
    )

    csv_path=os.path.join(get_abs_path(),'static','tmp2','kmeans.csv')
    cluster_data.to_csv(csv_path)
    return render_template('d3.html',data_file=url_for('static',filename='tmp2/kmeans.csv'))

@app.route('/d3_2')
def d3_2():
    df=get_data()
    X=df.ix[:, (df.columns !='class') & (df.columns !='code')].as_matrix() #this gives a numpy
    scatter_data=pd.DataFrame(
        {
         'x':X[:,0],
         'y':X[:,1],
         'z':X[:,2]
        }
    )
    csv_path=os.path.join(get_abs_path(),'static','tmp2','3D.csv')
    scatter_data.to_csv(csv_path)
    return render_template('d3_2.html',data=url_for('static',filename='tmp2/3D.csv'))

@app.route('/prediction')
def prediction():
    df=get_data()
    X=df.ix[:, (df.columns !='class') & (df.columns !='code')].as_matrix() #this gives a numpy
    print X
    y1=df.ix[:,df.columns=='class'].as_matrix()
    y=y1.reshape(683,1)
    Y=binary(y)
    #split into test and training sets

    features_train, features_test, outcome_train, outcome_test = cv.train_test_split(X, Y, test_size=0.4,random_state=1)

    #Random Forest Classifier
    forest=RandomForestClassifier(n_estimators=10,min_samples_leaf= 10, criterion=
    'gini', max_features= 'auto', max_depth= None)
    forest=forest.fit(features_train,outcome_train)
    predicted=forest.predict(features_test)


    #ROC Curve
    fig=plt.figure()
    predicted_score=forest.predict_proba(features_test)
    print predicted_score
    fpr,tpr,thresholds=metrics.roc_curve(outcome_test,predicted_score[:,1])
    roc_auc=metrics.auc(fpr,tpr)
    print roc_auc
    plt.plot(fpr, tpr, label='ROC curve (area = %0.1f)' % roc_auc, lw=2, color ="#0000ff")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.005, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")


    fig_path=os.path.join(get_abs_path(),'static','tmp2','roc.png')
    fig.savefig(fig_path)
    return render_template('prediction.html',fig=url_for('static',filename='tmp2/roc.png'))

@app.route('/api/v1/prediction_confusion_matrix')
def prediction_confusion_matrix():
    df=get_data()
    X=df.ix[:, (df.columns !='class') & (df.columns !='code')].as_matrix() #this gives a numpy
    print X
    y1=df.ix[:,df.columns=='class'].as_matrix()
    y=y1.reshape(683,1)
    Y=binary(y)
    #split into test and training sets

    features_train, features_test, outcome_train, outcome_test = cv.train_test_split(X, Y, test_size=0.4,random_state=1)

    #Random Forest Classifier for the classification.
    forest=RandomForestClassifier(n_estimators=10,min_samples_leaf= 10, criterion=
    'gini', max_features= 'auto', max_depth= None)
    forest=forest.fit(features_train,outcome_train)
    predicted=forest.predict(features_test)

    #Confusion_matrix
    confusion_matrix=metrics.confusion_matrix(outcome_test,predicted)

    output ={
          'Random Forest Classifier':

          {
            'fp':confusion_matrix[0][1],
            'tp':confusion_matrix[1][1],
            'fn':confusion_matrix[1][0],
            'tn':confusion_matrix[0][0]
          }
    }
    return jsonify(output)

@app.route('/head')
def head():
    df=get_data().head()
    data=json.loads(df.to_json())
    return jsonify(data)


