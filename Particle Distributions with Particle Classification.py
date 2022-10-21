import time
start = time.time()

from cmath import nan
import glob
import os
import math
from tokenize import Ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic

#Load in data sets:
#Data set from image analysis training data that includes manually classified particles
particle_char = pd.read_csv(r"YOUR PATH AND FILE HERE.csv")

#Data set from image analysis that included all particles (10 mesh) for raw anatomical fractions
files = os.path.join(r"YOUR PATH HERE", "FILE PREFIX*.csv")
# list of merged files returned
files = glob.glob(files)
# joining files with concat and read_csv
Mesh10_char = pd.concat(map(pd.read_csv, files), ignore_index=True)

#Define lists and variables for analysis and plotting:
anatomies = ['Pith','Rind']#['Sheath','Leaf','Pith','Husk','Rind','Cob']#,'Internode']#['Sheath','Leaf','Pith','Husk','Rind','Cob']#,'Internode'] #
ACspeeds = ['7.5 Hz','12.5 Hz','17.5 Hz','35 Hz','45 Hz']
Gravity_fractions = ['30-70','Fines','Gravity Heavy','Gravity Mid','Gravity Light','A2']
ProgEH_times = ['Raw','Raw Soaked','10% NaOH','30min','1hr','6hr','12hr','72hr'] #'24hr', ,
particle_types = [3,0,1,2]
particle_names = ['All particles','Bundle','Parenchyma','Fiber']
colors = ['b','g','xkcd:maize','xkcd:orange','r','xkcd:dark red','darkviolet','tab:brown']#['tab:brown','g','darkviolet','xkcd:light blue','r','b'] 'orange' ['b','g','yellow','organge','r','darkred']
interest = 'Major'#['Major','Minor','Feret','AR','Solidity','Circularity']
meshes = ['#10','#20']
mesh_size = ['2 mm','0.85 mm']
pretreatments = ['Raw','Raw Soaked','10% NaOH 90C']#['Raw','10% NaOH 90C','LHW190'] #'LHW190'
pretreatment_names = ['Raw',('LHW 190' + u'\u2103')]#['Raw',('10% NaOH 90' + u'\u2103'),('LHW 190' + u'\u2103')]
X_names = ['Parent','Pretreatment','Anatomy','Mesh Size','Area','Perim','Width','Height','Major','Minor','Circularity','Feret','MinFeret','AR','Round','Solidity','%Area','ParticleType']
markers = ['o','s','v','^','D','p','>','<']  #['o','s','v','^','D','p','>','<']
bins_num = 25  #bins for length = 600, AR= 300, Roundness = 35, width = 300, minFeret = 400, solidity = 55, % Area = 35

#Options to modify data sets to disclude particles with high circularity (bubbles) or fines (area < X)

drop_circularity = 0.98
drop_roundness = 0.96
drop_area = 0.002
upper_fines_limit = 0.003

particle_char = particle_char.drop(particle_char[(particle_char['Area'] < drop_area)].index) #Drops particles that are small
particle_char = particle_char.drop(particle_char[(particle_char['Circularity'] > drop_circularity)].index) #Drops particles that are round (bubbles)

#make list of all particles
All_Particles = pd.concat([Mesh10_char],ignore_index=True) 
#dfs = [particle_char,Mesh10_char,Mesh20_char,Air_class_char,LHW190_10Mesh,ProgEH]
#particle_char = pd.concat(dfs, ignore_index=True)

#Add "Raw" as the pretreatment for raw samples
All_Particles['Pretreatment'] = All_Particles['Pretreatment'].fillna('Raw')
#Get only columns pertinent to training and remove rows with NaN
All_Particles = All_Particles[X_names]
All_Particles = All_Particles.drop(columns=['ParticleType'])
All_Particles = All_Particles.dropna()

All_Particles = All_Particles.drop(All_Particles[(All_Particles['Area'] < drop_area)].index) #Drops particles that are small
All_Particles = All_Particles.drop(All_Particles[(All_Particles['Circularity'] > drop_circularity)].index) #Drops particles that are circular (bubbles)
All_Particles = All_Particles.drop(All_Particles[(All_Particles['Round'] > drop_roundness)].index) #Drops particles that are round (bubbles)
#All_Particles = All_Particles.drop(All_Particles[(All_Particles['%Area'] == 100)].index)

def fines():
    #Calculate percent fines
    fines_table_index = ['10 Mesh','20 Mesh','10 Mesh LHW190']
    AC_fines_index = ['Air Classified']

    pct_fines_all10 = []
    for anat in anatomies:
        Mesh10_loop = Mesh10_char[Mesh10_char['Anatomy'] == anat]
        fines_area = Mesh10_loop.loc[Mesh10_loop['Area']<upper_fines_limit,'Area'].sum()
        total_area = Mesh10_loop['Area'].sum()
        pct_fines = 100*fines_area/total_area
        pct_fines_all10 = np.append(pct_fines_all10, pct_fines)

    pct_fines_all20 = []
    for anat in anatomies:
        Mesh20_loop = Mesh20_char[Mesh20_char['Anatomy'] == anat]
        fines_area = Mesh20_loop.loc[Mesh20_loop['Area']<upper_fines_limit,'Area'].sum()
        total_area = Mesh20_loop['Area'].sum()
        pct_fines = 100*fines_area/total_area
        pct_fines_all20 = np.append(pct_fines_all20, pct_fines)

    pct_fines_allLHW190 = []
    for anat in anatomies:
        LHW190_loop = LHW190_10Mesh[LHW190_10Mesh['Anatomy'] == anat]
        fines_area = LHW190_loop.loc[LHW190_loop['Area']<upper_fines_limit,'Area'].sum()
        total_area = LHW190_loop['Area'].sum()
        pct_fines = 100*fines_area/total_area
        pct_fines_allLHW190 = np.append(pct_fines_allLHW190, pct_fines)

    pct_fines_All_Particles = []
    for PEHtime in ProgEH_times:
        AP_loop = All_Particles[(All_Particles['Parent'].str.contains(PEHtime,na=False)) & (All_Particles['Anatomy'] == 'Rind')]
        fines_area = AP_loop.loc[AP_loop['Area']<upper_fines_limit,'Area'].sum()
        total_area = AP_loop['Area'].sum()
        pct_fines = 100*fines_area/total_area
        pct_fines_All_Particles = np.append(pct_fines_All_Particles, pct_fines)

    pct_fines_table = pd.DataFrame(data = [pct_fines_all10,pct_fines_all20,pct_fines_allLHW190], columns = anatomies, index = fines_table_index)
    print(pct_fines_table)

    pct_fines_all_AirClass = []
    for speed in ACspeeds:
        AirClass_loop = particle_char[particle_char['Anatomy'] == speed]
        fines_area = AirClass_loop.loc[AirClass_loop['Area']<upper_fines_limit,'Area'].sum()
        total_area = AirClass_loop['Area'].sum()
        pct_fines = 100*fines_area/total_area
        pct_fines_all_AirClass = np.append(pct_fines_all_AirClass, pct_fines)

    pct_fines_table_AC = pd.DataFrame(data = [pct_fines_all_AirClass], columns = ACspeeds, index = AC_fines_index)
    print(pct_fines_table_AC)
#fines()

def training():
    S_count = len(particle_char[(particle_char['ParticleType']=='Shive')])
    P_count = len(particle_char[(particle_char['ParticleType']=='Parenchyma')])
    F_count = len(particle_char[(particle_char['ParticleType']=='Fiber')])
    min_count = min(S_count,P_count,F_count)
    Xall = particle_char[X_names]
    FiberX = Xall[(Xall['ParticleType']=='Fiber')]
    FiberXsub = FiberX.sample(n=min_count)
    ParenchymaX = Xall[(Xall['ParticleType']=='Parenchyma')]
    ParenchymaXsub = ParenchymaX.sample(n=min_count)
    ShiveX = Xall[(Xall['ParticleType']=='Shive')]
    ShiveXsub = ShiveX.sample(n=min_count)
    X = pd.concat((FiberXsub,ParenchymaX,ShiveXsub))
    y = X['ParticleType']
    X = X.drop(columns=['Parent','ParticleType','Mesh Size','Anatomy','Pretreatment'])
    y = pd.DataFrame(y)
    y.loc[y['ParticleType'] == 'Shive', 'TypeTeritiary'] = 0 
    y.loc[y['ParticleType'] == 'Parenchyma', 'TypeTeritiary'] = 1
    y.loc[y['ParticleType'] == 'Fiber', 'TypeTeritiary'] = 2
    y = y.drop(columns=['ParticleType'])
    X = np.array(X)
    y = np.ravel(np.array(y))
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=24)
    name = "Gaussian Process"
    global clf 
    clf = GaussianProcessClassifier(1.0 * RationalQuadratic())
    clf.fit(X_train, np.ravel(y_train))
    # y_pred = clf.predict(X_test)

    # #Generate table of scores
    # score = clf.score(X_test, y_test)
    # clf_score = score
    # percentage = "{:.1%}". format(score)
    # clf_percent = percentage
training()

def predict():
    # X_pred = All_Particles.drop(columns=['Parent','Pretreatment','Anatomy','Mesh Size'])
    # X_pred = StandardScaler().fit_transform(X_pred)
    # y_pred = clf.predict(X_pred)
    # All_Particles['PredParticle'] = y_pred

    #Option for large arrays (divide in sets and predict separately):
    X_pred = All_Particles.drop(columns=['Parent','Pretreatment','Anatomy','Mesh Size'])
    X_pred = StandardScaler().fit_transform(X_pred)
    chop = math.ceil(len(X_pred)/350000)
    y_pred = []
    for i in range(1, chop+1):
        if i == chop:
            X_pred1 = X_pred[(i-1)*350000:,:]
        else:
            X_pred1 = X_pred[(i-1)*350000:(350000*i),:]
        y_pred1 = clf.predict(X_pred1)
        y_pred = np.concatenate((y_pred,y_pred1))
    All_Particles['PredParticle'] = y_pred
predict()

pd.DataFrame(All_Particles).to_csv(r'YOUR PATH AND FILE HERE.csv',index=False)

def particle_count():
    for i,PEHtime in enumerate(ProgEH_times):
        AP_loop = All_Particles.drop(All_Particles[(All_Particles['Area'] < drop_area)].index)
        AP_loop = AP_loop[(AP_loop['Parent'].str.contains(PEHtime,na=False)) &  (AP_loop['Anatomy'] == 'Rind')]
        all_count =  len(AP_loop)
        bundle_count = len(AP_loop[(AP_loop['PredParticle']==0)])
        parenchyma_count = len(AP_loop[(AP_loop['PredParticle']==1)])
        fiber_count = len(AP_loop[(AP_loop['PredParticle']==2)])
        counts = np.array([[all_count],[bundle_count],[parenchyma_count],[fiber_count]])
        if i == 0:
            count_All_Particles = counts
        else:
            count_All_Particles = np.hstack((count_All_Particles, counts))
#particle_count()

#def particle_area():
for i,gf in enumerate(Gravity_fractions):
    AP_loop = All_Particles.drop(All_Particles[(All_Particles['Area'] < drop_area)].index)
    AP_loop = AP_loop[(AP_loop['Anatomy'].str.contains(gf,na=False))]# &  (AP_loop['Anatomy'] == 'Rind')]
    all_area =  AP_loop['Area'].sum()
    bundle_area = AP_loop.loc[AP_loop['PredParticle']==0,'Area'].sum()
    parenchyma_area = AP_loop.loc[AP_loop['PredParticle']==1,'Area'].sum()
    fiber_area = AP_loop.loc[AP_loop['PredParticle']==2,'Area'].sum()
    areas = np.array([[all_area],[bundle_area],[parenchyma_area],[fiber_area]])
    if i == 0:
        area_All_Particles_gravity = areas
    else:
        area_All_Particles_gravity = np.hstack((area_All_Particles_gravity, areas))
pd.DataFrame(area_All_Particles_gravity).to_csv(r'C:\Users\Dylan\Documents\Montana State Hodge Group Research\Experimental\DOE INL Corn Stover\Particle Analysis (Valmet)\Particle Characteristic CSVs\Asif Pith-Rind April 2022\particle areas gravity - DA = 0.002.csv',index=False)
#particle_area()

#f, axes = plt.subplots(nrows=2,ncols=int(len(names)/2),figsize=(17,8),sharey='row')
#All_Particles.to_csv(r"C:\Users\Dylan\Documents\Montana State Hodge Group Research\Experimental\DOE INL Corn Stover\Particle Analysis (Valmet)\Python\Particle Distribution Analysis\10Mesh_with_prediction.csv")

#loop through particle types

def progEH_plotting():
    for part_type, part_name in zip(particle_types,particle_names):
        ProgEH_times = ['Raw']#['30-70','Fines','Gravity Heavy','Gravity Mid','Gravity Light','A2']#['Raw','10% NaOH','30min','1hr','6hr','12hr','72hr'] #'Raw Soaked', ['7.5 Hz','12.5 Hz','17.5 Hz','35 Hz','45 Hz']
        colors = ['b','g','xkcd:maize','xkcd:orange','r','xkcd:dark red','darkviolet']#['tab:brown','g','darkviolet','xkcd:light blue','r','b'] 'orange' ['b','g','yellow','organge','r','darkred']
        markers = ['o','s','v','^','D','p','>']
        bins_num = 25
        interest = '%Area'
        drop_area = 0.01

        char = All_Particles[[interest,'Parent','Pretreatment','Anatomy','Mesh Size','Area','PredParticle']]
        char = char.drop(char[(char['Area'] < drop_area)].index) #Drops particles that are small
        char = char.drop(char[(char['Parent'].str.contains('Raw Soaked',na=False))].index)
        values_array = {}
        fig = plt.figure()
        ax = fig.add_subplot()

        idx = 0

        
        for c,EHtime,m in zip(colors,ProgEH_times,markers):

            #In following line, define what subset of pretreatments and particles to investigate/plot
            if part_type == 3:
                char_anat = char[(char['Anatomy'].str.contains(EHtime, na=False))]# & (char['Anatomy'] == 'Rind')] # & (char['Round'] <= 0.333)]    
            else:
                char_anat = char[(char['Anatomy'].str.contains(EHtime, na=False)) & (char['PredParticle'] == part_type)]  #& (char['Round'] <= 0.333) # & (char['Anatomy'] == 'Rind')
                    
            char_anat_loop = char_anat
            
            char_max = char_anat_loop[interest].max()
            char_min = char_anat_loop[interest].min()
            char_range = char_max-char_min
            
            if interest == 'Major' or interest == 'Area' or interest == 'MinFeret' or interest == 'Minor':
                bins_val = np.logspace(start=np.log10(char_min), stop=np.log10(char_max), num=bins_num)
            else:
                bins_val = np.linspace(start=char_min, stop=char_max, num=bins_num)

            if char_anat_loop.empty:
                continue
            else:
                if interest == 'Major' or interest == 'Area' or interest == 'MinFeret' or interest == 'Minor':
                    hist = np.histogram(char_anat_loop[interest], bins=np.logspace(start=np.log10(char_min), stop=np.log10(char_max), num=bins_num+1), range=(char_min,char_max), weights = char_anat_loop['Area'], density=False)
                else: 
                    hist = np.histogram(char_anat_loop[interest], bins=np.linspace(start=char_min, stop=char_max, num=bins_num+1), range=(char_min,char_max), weights = char_anat_loop['Area'], density=False)
                values = hist[0]
                values = 100*values/sum(values)
                if idx == 0:
                    max_y = values.max()
                else:
                    if values.max() > max_y:
                        max_y = values.max()
                plt.plot(bins_val,values,marker=m,color=c,markeredgecolor='black',ls='-',label=('Air classified - ' + '%s' %EHtime)) #('%s - ' %anat + '%s' %ptn))
                            
                if interest == 'Major' or interest == 'Area' or interest == 'MinFeret' or interest == 'Minor':
                    plt.xlim(0.01,1)
                    plt.xticks(np.arange(0.01, 10.1, 1))
                    plt.xscale('log', base=10)
                elif interest == '%Area':
                    plt.xlim(80,100)
                    plt.xticks(np.arange(80, 100.1, 5))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                else:
                    plt.xlim(0,1)
                    plt.xticks(np.arange(0.0, 1.01, 0.2))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                            
                plt.ylim(-0.3,18)
                plt.yticks(np.arange(0, 18.1, 3))
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
                values_array = np.append(values_array,sum(values))

            idx += 1

        if interest == '%Area' or interest == 'Solidity':
            plt.legend(prop=font_manager.FontProperties(weight='bold', size = 10),loc='upper left')
        else:
            plt.legend(prop=font_manager.FontProperties(weight='bold', size = 10),loc='upper right')

        if interest == 'Major':
            plt.xlabel('Length (mm)',fontweight = 'bold',size=16,labelpad=15)
        elif interest == 'Round':
            plt.xlabel('Roundness',fontweight = 'bold',size=16,labelpad=15)
        elif interest == 'Minor':
            plt.xlabel('Width (mm)',fontweight = 'bold',size=16,labelpad=15)
        elif interest == 'Solidity':
            plt.xlabel('Solidity',fontweight = 'bold',size=16,labelpad=15)
        elif interest == '%Area':
            plt.xlabel('%Area',fontweight = 'bold',size=16,labelpad=15)
        elif interest == 'Area':
            plt.xlabel('Area ($\mathregular{mm^2}$)',fontweight = 'bold',size=16,labelpad=15)
        elif interest == 'MinFeret':
            plt.xlabel('Minimum Feret Diameter (mm)',fontweight = 'bold',size=16,labelpad=15) 

        plt.ylabel('Probability', fontweight = 'bold',size=16,labelpad=15)
        plt.xticks(fontsize= 12, fontweight = 'bold')
        plt.yticks(fontsize= 12, fontweight = 'bold')

        ax.tick_params(direction ='in', width = 1.25)
        ax.tick_params(which='minor',direction ='in',width = 1.25)
        ax.xaxis.set_label_coords(.5, -0.1)
        if interest == 'Major':
            ax.set_xscale('log')
        sides = ['top','bottom','left','right']
        for x in sides:
            ax.spines[x].set_linewidth(1.25)
        plt.title(part_name,fontweight = 'bold',size=16)
        plt.tight_layout()

        plt.show()



def distribution_plotting():

    char = All_Particles[[interest,'Parent','Pretreatment','Anatomy','Mesh Size','Area']]
    char = char.drop(char[(char['Area'] < drop_area)].index) #Drops particles that are small
    values_array = {}
    fig = plt.figure()
    ax = fig.add_subplot()

    idx = 0

    for anat,c in zip(anatomies,colors):
        for mesh,ms,m in zip(meshes,mesh_size,markers):
        
            #In following line, define what subset of pretreatments and particles to investigate/plot
            #if part_type == 3:
            char_anat = char[(char['Anatomy'].str.contains(anat, na=False)) & (char['Pretreatment'] == 'Raw')] #& (char['Mesh Size'] == mesh)]    
            #else:
            #   char_anat = char[(char['Anatomy'].str.contains(anat, na=False)) & (char['Mesh Size'] == mesh) & (char['Pretreatment'] == 'Raw') & (char['PredParticle'] == part_type)]
            
            char_anat_loop = char_anat

            char_max = char_anat_loop[interest].max()
            char_min = char_anat_loop[interest].min()
            char_range = char_max-char_min
            
            if interest == 'Major' or interest == 'Area' or interest == 'Perim':
                bins_val = np.logspace(start=np.log10(char_min), stop=np.log10(char_max), num=bins_num)
            else:
                bins_val = np.linspace(start=char_min, stop=char_max, num=bins_num)

            if char_anat_loop.empty:
                continue
            else:
                if interest == 'Major' or interest == 'Area' or interest == 'Perim':
                    hist = np.histogram(char_anat_loop[interest], bins=np.logspace(start=np.log10(char_min), stop=np.log10(char_max), num=bins_num+1), range=(char_min,char_max), weights = char_anat_loop['Area'], density=False)
                else: 
                    hist = np.histogram(char_anat_loop[interest], bins=np.linspace(start=char_min, stop=char_max, num=bins_num+1), range=(char_min,char_max), weights = char_anat_loop['Area'], density=False)
                values = hist[0]
                values = 100*values/sum(values)
                if idx == 0:
                    max_y = values.max()
                else:
                    if values.max() > max_y:
                        max_y = values.max()
                plt.plot(bins_val,values,marker=m,color=c,markeredgecolor='black',ls='-',label=('%s - '%anat + '%s' %ms))
                            
                if interest == 'Major' or interest == 'Area' or interest == 'Perim':
                    plt.xlim(0.001,7)
                    plt.xticks(np.arange(0.03, 7.1, 1))
                    plt.xscale('log', base=10)
                elif interest == '%Area':
                    plt.xlim(85,100)
                    plt.xticks(np.arange(60, 100.1, 5))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                else:
                    plt.xlim(0,1)
                    plt.xticks(np.arange(0.0, 1.01, 0.2))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                            
                plt.ylim(-0.25,10)
                plt.yticks(np.arange(0, 18.1, 3))
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
                values_array = np.append(values_array,sum(values))

            idx += 1

    plt.legend(prop=font_manager.FontProperties(weight='bold', size = 10),loc='upper left')

    if interest == 'Major':
        plt.xlabel('Length (mm)',fontweight = 'bold',size=16,labelpad=15)
    elif interest == 'Round':
        plt.xlabel('Roundness',fontweight = 'bold',size=16,labelpad=15)
    elif interest == 'Solidity':
        plt.xlabel('Solidity',fontweight = 'bold',size=16,labelpad=15)
    elif interest == '%Area':
        plt.xlabel('%Area',fontweight = 'bold',size=16,labelpad=15)
    elif interest == 'Area':
        plt.xlabel('Area (mm^2)',fontweight = 'bold',size=16,labelpad=15)
    elif interest == 'Perim':
        plt.xlabel('Perimeter (mm)',fontweight = 'bold',size=16,labelpad=15)

    plt.ylabel('Probability', fontweight = 'bold',size=16,labelpad=15)
    plt.xticks(fontsize= 12, fontweight = 'bold')
    plt.yticks(fontsize= 12, fontweight = 'bold')

    ax.tick_params(direction ='in', width = 1.25)
    ax.tick_params(which='minor',direction ='in',width = 1.25)
    ax.xaxis.set_label_coords(.5, -0.1)
    if interest == 'Major':
        ax.set_xscale('log')
    sides = ['top','bottom','left','right']
    for x in sides:
        ax.spines[x].set_linewidth(1.25)
    #plt.title(part_name,fontweight = 'bold',size=16)
    plt.tight_layout()

    plt.show()


end = time.time()
print(end-start)
#plt.savefig(r'C:\Users\Dylan\Documents\Montana State Hodge Group Research\Papers\Particle Analysis Paper\PCA.png', dpi=600)

print('All done!')


