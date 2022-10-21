import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler


np.random.seed(5)

particle_names = ['Bundle','Parenchyma','Fiber']
X_names = ['Area','Perim','Width','Height','Major','Minor','Circularity','Feret','MinFeret','AR','Round','Solidity','%Area','ParticleType']
particle_char = pd.read_csv(r"YOUR PATH AND FILE HERE.csv")
particle_char = particle_char[(particle_char['Circularity'] < 0.95) & (particle_char['Area']>0.003)]
Xall = particle_char[X_names]
S_count = len(particle_char[(particle_char['ParticleType']=='Shive')])
P_count = len(particle_char[(particle_char['ParticleType']=='Parenchyma')])
F_count = len(particle_char[(particle_char['ParticleType']=='Fiber')])
min_count = min(S_count,P_count,F_count)

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

fig = plt.figure()
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
plt.cla()
pca = decomposition.PCA()
pca.fit(X)
X = pca.transform(X)
var0 = round(pca.explained_variance_ratio_[0]*100,1)
var1 = round(pca.explained_variance_ratio_[1]*100,1)
var2 = round(pca.explained_variance_ratio_[2]*100,1)

legend_properties = {'weight':'bold'}


for name,lab,color,m in [("Bundle", 0,'g','o'), ("Parenchyma", 1,'b','s'), ("Fiber", 2,'r','^')]:
    
    ax.scatter(X[y == lab, 0], X[y == lab, 1], X[y == lab, 2], c=color, marker=m, edgecolor="k", label=name)
    ax.legend(name,prop=legend_properties)


ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('PC1 ' + '(' + str(var0) + ')' + '%',fontweight='bold',fontsize='large')
ax.set_ylabel('PC2 ' + '(' + str(var1) + ')' + '%',fontweight='bold',fontsize='large')
ax.set_zlabel('PC3 ' + '(' + str(var2) + ')' + '%',fontweight='bold',fontsize='large')

ax.legend(loc=10, bbox_to_anchor=(0.15, 0.6),fontsize='large')

ax.grid(False)
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis._axinfo['tick']['inward_factor'] = 0
ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
ax.yaxis._axinfo['tick']['inward_factor'] = 0
ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
ax.zaxis._axinfo['tick']['inward_factor'] = 0
ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

plt.show()

print('hello')