import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import decomposition

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

particle_names = ['Bundle','Parenchyma','Fiber']
X_names = ['Area','Perim','Width','Height','Major','Minor','Circularity','Feret','MinFeret','AR','Round','Solidity','%Area','ParticleType']
#'Area','Perim','Width','Height','Major','Minor','Circularity','Feret','MinFeret','AR','Round','Solidity'
particle_char = pd.read_csv(r"YOUR PATH AND FILE HERE.csv")
particle_char = particle_char[(particle_char['Circularity'] < 0.8) & (particle_char['Area']>0.01)]
#X = particle_char.drop(columns=['Sample','ParticleNumber','Angle','FeretX','FeretY','FeretAngle','ParticleClass','Anatomy','ParticleType','','RawGlucoseYield','Alk90-10PTYield','Glucan','Xylan','Lignin','Acetate','Ash','WRV','ASL','WaterExt','EtOHExt','Moisture'])
Xall = particle_char[X_names]
S_count = len(particle_char[(particle_char['ParticleType']=='Shive')])
P_count = len(particle_char[(particle_char['ParticleType']=='Parenchyma')])
F_count = len(particle_char[(particle_char['ParticleType']=='Fiber')])
min_count = min(S_count,P_count,F_count)

f, axes = plt.subplots(nrows=2,ncols=int(len(names)/2),figsize=(17,8),sharey='row')

loops = 30
splits = list(range(0,loops+1))
clf_scores_total = []

#Iterate over loops train/test splits with new data sampled in each j loop
for j in splits:
    FiberX = Xall[Xall['ParticleType']=='Fiber']
    FiberXsub = FiberX.sample(n=min_count)
    ParenchymaX = Xall[Xall['ParticleType']=='Parenchyma']
    ParenchymaXsub = ParenchymaX.sample(n=min_count)
    ShiveX = Xall[Xall['ParticleType']=='Shive']
    ShiveXsub = ShiveX.sample(n=min_count)
    X = pd.concat((FiberXsub,ParenchymaXsub,ShiveXsub))
    y = X['ParticleType']
    X = X.drop(columns=['ParticleType'])
    y = pd.DataFrame(y)
    y.loc[y['ParticleType'] == 'Shive', 'TypeTeritiary'] = 0 
    y.loc[y['ParticleType'] == 'Parenchyma', 'TypeTeritiary'] = 1
    y.loc[y['ParticleType'] == 'Fiber', 'TypeTeritiary'] = 2
    y = y.drop(columns=['ParticleType'])
    X = np.array(X)
    y = np.ravel(np.array(y))
    X = StandardScaler().fit_transform(X)
    
    #Option for PCA on X prior to training... doesn't seem to improve average accuracies
    # pca = decomposition.PCA()
    # pca.fit(X)
    # X = pca.transform(X)
    # X = X[:,0:3]
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
          
    # iterate over classifiers
    i=0
    for name, clf in zip(names, classifiers):
        clf_percent = []
        clf.fit(X_train, np.ravel(y_train))
        y_pred = clf.predict(X_test)
        
        #Generate table of scores
        score = clf.score(X_test, np.ravel(y_test))
        if i == 0:
            clf_scores = score
            percentage = "{:.1%}". format(score)
            clf_percent = percentage
        else:
            percentage = "{:.1%}". format(score)
            clf_scores = np.append(clf_scores,score)
            clf_percent = np.append(clf_percent,percentage)
        

        if j == loops:
            #Generate confusion matricies and plot them in subplot
            if i < len(names)/2:
                cm = confusion_matrix(y_test,y_pred,labels=clf.classes_)
                disp = ConfusionMatrixDisplay(cm, display_labels=particle_names) 
                disp.plot(ax=axes[0,i],cmap=plt.cm.Blues,xticks_rotation=45)
                disp.ax_.set_title(name + "\n" + str(percentage))
                disp.im_.colorbar.remove()
                disp.ax_.set_xlabel('')
                if i!=0:
                    disp.ax_.set_ylabel('')
                else:
                    disp.ax_.set_ylabel('Actual',size=18)
                
            else:
                cm = confusion_matrix(y_test,y_pred,labels=clf.classes_)
                disp = ConfusionMatrixDisplay(cm, display_labels=particle_names) 
                disp.plot(ax=axes[1,i-int(len(names)/2)],cmap=plt.cm.Blues,xticks_rotation=45)
                disp.ax_.set_title(name + "\n" + str(percentage))
                disp.im_.colorbar.remove()
                disp.ax_.set_xlabel('')
                if i==int(len(names)/2):
                    disp.ax_.set_ylabel('Actual',size=18)
                else:
                    disp.ax_.set_ylabel('')
        
        i += 1 

    if j==0:
        clf_scores_total = np.transpose(clf_scores)
    else:
        clf_scores_total = np.vstack([clf_scores_total,np.transpose(clf_scores)])

average_clf_score = np.mean(clf_scores_total,axis=0)
std_clf_score = np.std(clf_scores_total,axis=0)
np.savetxt("AverageScore withPCA.csv", average_clf_score, delimiter=",")
np.savetxt("SDScore withPCA.csv", std_clf_score, delimiter=",")

f.text(0.4, 0.05, 'Predicted', ha='left',size = 18)
plt.subplots_adjust(top=1.0, bottom=0.11, left=0.08, right=0.955, hspace=0.1, wspace=0.285)
f.colorbar(disp.im_, ax=axes, pad=0.1)
plt.show()

print('hello')
