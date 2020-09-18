import numpy as np
import itertools
import math
from utils import readCsv
from readNoduleList import joinNodules

def calcCTFleischnerProb(nd,nn,v0,v1,v2,t0,t1,t2,verb=False):
    # Compute probability of belonging to each Fleischner class if given lists with:
    # nd: the probability of each finding being a nodule
    # nn: 1-nd
    # v0-v2: the probability of each finding belonging to each volume class
    # t0-t2: the probability of each finding belonging to each texture class
    p0 = 0.0
    p1 = 0.0
    p2 = 0.0
    p3 = 0.0
    
    vs = np.sum([v0,v1,v2],axis=0)
    ts = np.sum([t0,t1,t2],axis=0)
    v0=v0/vs
    v1=v1/vs
    v2=v2/vs
    t0=t0/ts
    t1=t1/ts
    t2=t2/ts
    
    # No nodules
    p0 += np.prod(nn)
    if verb:
        print('none',p0,p1,p2,p3)
    
    # Single nodule
    for i in range(len(nd)):
        p0 += np.prod(np.delete(nn,i))*nd[i]*v0[i]
        p1 += np.prod(np.delete(nn,i))*nd[i]*((v1[i]+v2[i])*t0[i]+v1[i]*t2[i])
        p2 += np.prod(np.delete(nn,i))*nd[i]*((v1[i]+v2[i])*t1[i])
        p3 += np.prod(np.delete(nn,i))*nd[i]*v2[i]*t2[i]
    if verb:
        print('single',p0,p1,p2,p3)
    
    # Multiple nodules
    if len(nd)>1:
        combN = []
        indN = [i for i in range(len(nd))]
        for i in range(2,len(nd)+1):
            combN.extend(list(itertools.combinations(indN,i)))
        combN = [list(c) for c in combN]
        for cN in combN:
            cNN = [i for i in indN if not i in cN]
            # All Non Solid
            pans = np.prod(nd[cN]*(t0[cN]+t1[cN]))*np.prod(nn[cNN])
            p3 += pans
            if verb:
                print('pans',p0,p1,p2,p3)
            # All Solid
            pas = np.prod(nd[cN]*t2[cN])*np.prod(nn[cNN])
            pasp0 = np.prod(nd[cN]*t2[cN]*v0[cN])*np.prod(nn[cNN])
            combV2 = []
            for i in range(1,len(cN)+1):
                combV2.extend(list(itertools.combinations(cN,i)))
            pasp3 = 0.0
            
            combV2 = [list(c) for c in combV2]
            for cV2 in combV2:
                cV01 = [i for i in cN if not i in cV2]
                pasp3 += np.prod(nd[cV2]*t2[cV2]*v2[cV2])*np.prod(nd[cV01]*t2[cV01]*(v0[cV01]+v1[cV01]))*np.prod(nn[cNN])
            p0 += pasp0
            p3 += pasp3
            p2 += pas-pasp3-pasp0
            if verb:
                print('pas',p0,p1,p2,p3)
            # Mixed
            combT2 = []
            for i in range(1,len(cN)):
                combT2.extend(list(itertools.combinations(cN,i)))
            combT2 = [list(c) for c in combT2]
            for cT2 in combT2:
                cT01 = [i for i in cN if not i in cT2]
                pcomb = np.prod(nn[cNN])*np.prod(nd[cN])*np.prod(t2[cT2])*np.prod(t0[cT01]+t1[cT01])
                pT2 = calcCTFleischnerProb_Mixed(v0[cT2],v1[cT2],v2[cT2],t0[cT2],t1[cT2],t2[cT2],'allsolid')
                pT01 = calcCTFleischnerProb_Mixed(v0[cT01],v1[cT01],v2[cT01],t0[cT01],t1[cT01],t2[cT01],'allnonsolid')
                p0 += pT2[0]*pT01[0]*pcomb
                p1 += (pT2[1]*pT01[0]+pT2[0]*pT01[1]+pT2[1]*pT01[1])*pcomb
                p2 += (pT2[2]*np.sum(pT01[:2])+np.sum(pT2[:2])*pT01[2]+pT2[2]*pT01[2])*pcomb
                p3 += (pT2[3]*np.sum(pT01[:3])+np.sum(pT2[:3])*pT01[3]+pT2[3]*pT01[3])*pcomb
            if verb:
                print('mxd',p0,p1,p2,p3)
    
    if verb:
        print('sum',p0+p1+p2+p3)
    return [p0,p1,p2,p3]

def calcCTFleischnerProb_Mixed(v0,v1,v2,t0,t1,t2,texstr,verb = False):
    # Compute probability of belonging to each Fleischner class for mixed nodule lists (called by calcCTFleischnerProb):
    p0 = 0.0
    p1 = 0.0
    p2 = 0.0
    p3 = 0.0
    
    if texstr=='allsolid':
        t0[:]=0
        t1[:]=0
        t2[:]=t2/t2
        t2[np.where(np.isnan(t2))] = 0
    else:
        ts = t0+t1
        t0=t0/ts
        t1=t1/ts        
        t0[np.where(np.isnan(t0))] = 0
        t1[np.where(np.isnan(t1))] = 0
        t2[:]=0
    
    if len(v0)==1: # Single nodule
        p0 += v0[0]
        p1 += (v1[0]+v2[0])*t0[0]+v1[0]*t2[0]
        p2 += (v1[0]+v2[0])*t1[0]
        p3 += v2[0]*t2[0]
        if verb:
            print('single mixed',p0,p1,p2,p3)
    
    elif len(v0)>1: # Multiple nodules
        # All Non Solid
        pans = np.prod((t0+t1))
        p3 += pans
        if verb:
            print('pans mixed',p0,p1,p2,p3)
        # All Solid
        pas = np.prod(t2)
        pasp0 = np.prod(t2*v0)
        
        combV2 = []
        indN = [i for i in range(len(v0))]
        for i in range(1,len(v0)+1):
            combV2.extend(list(itertools.combinations(indN,i)))
        pasp3 = 0.0
        
        combV2 = [list(c) for c in combV2]
        for cV2 in combV2:
            cV01 = [i for i in indN if not i in cV2]
            pasp3 += np.prod(t2[cV2]*v2[cV2])*np.prod(t2[cV01]*(v0[cV01]+v1[cV01]))
        p0 += pasp0
        p3 += pasp3
        p2 += pas-pasp3-pasp0
        if verb:
            print('pas mixed',p0,p1,p2,p3)
    
    return [p0,p1,p2,p3]

def calcCTFleischnerScore(nd,vclass,tclass,nodprob = 0.5):
    # Compute Fleischner class if given lists with:
    # nd: the probability of each finding being a nodule
    # vclass: each finding's volume class (0,1,2)
    # tclass: each finding's texture class (0,1,2)
    # nodprob: threshold to use to divide nodules/nonnodules
    vclass = vclass[nd>=nodprob]
    tclass = tclass[nd>=nodprob]
    nd = nd[nd>=nodprob]
    
    if len(nd)==0:# No nodules
        f = 0
    elif len(nd)==1: # Single nodule
        if vclass[0]==0: #small
            f = 0
        elif tclass[0]==1: #partsolid
            f = 2        
        elif tclass[0]==0 or vclass[0]==1: #ggo/solid medium
            f = 1
        else:
            f = 3 #large
    else: # Multiple nodules
        if np.all(tclass<2): #all nonsolid
            f = 2
        elif np.all(tclass==2): #all solid
            if np.all(vclass==0): #small
                f = 0
            elif np.any(vclass==2): #at least one big
                f = 3
            else: #otherwise
                f = 2
        else: #mixed nodules
            fs = calcCTFleischnerScore(nd[tclass==2],vclass[tclass==2],tclass[tclass==2])
            fns = calcCTFleischnerScore(nd[tclass<2],vclass[tclass<2],tclass[tclass<2])
            f = max(fs,fns)
        
    return f

def calcFleischner(nodules):
    # Compute Fleischner class for a list of nodules from different CTs (trainNodules_gt.csv):
    # nodules: list of nodules read from csv
    header = nodules[0]
    lines = nodules[1:]  
    
    indlnd = header.index('LNDbID')
    LND = np.asarray([line[indlnd] for line in lines])
    
    if 'Volume0' in header and 'Text0' in header:
        # if probabilities of volume and texture class are given compute Fleischner class probabilities
        Nd = np.asarray([float(line[header.index('Nodule')]) for line in lines])
        Nn = 1-Nd
        V0 = np.asarray([float(line[header.index('Volume0')]) for line in lines])
        V1 = np.asarray([float(line[header.index('Volume1')]) for line in lines])
        V2 = np.asarray([float(line[header.index('Volume2')]) for line in lines])
        T0 = np.asarray([float(line[header.index('Text0')]) for line in lines])
        T1 = np.asarray([float(line[header.index('Text1')]) for line in lines])
        T2 = np.asarray([float(line[header.index('Text2')]) for line in lines])
    elif 'Volume' in header and 'Text' in header:
        # if volume and texture are given compute Fleischner class
        Nd = np.asarray([float(line[header.index('Nodule')]) for line in lines])
        V = np.asarray([float(line[header.index('Volume')]) for line in lines])
        T = np.asarray([float(line[header.index('Text')]) for line in lines])
    elif 'VolumeClass' in header and 'TextClass' in header:
        # if volume and texture class are given compute Fleischner class
        Nd = np.asarray([float(line[header.index('Nodule')]) for line in lines])
        Vclass = np.asarray([float(line[header.index('VolumeClass')]) for line in lines])
        Tclass = np.asarray([float(line[header.index('TextClass')]) for line in lines])
    
    p0 = 0; p1 = 0; p2 = 0; p3 = 0
    fleischner = [['LNDbID','Fleischner','Fleischner0','Fleischner1','Fleischner2','Fleischner3']]
    for lndU in np.unique(LND):
        if 'Volume0' in header and 'Text0' in header:
            # if probabilities of volume and texture class are given compute Fleischner class probabilities
            nd = Nd[LND==lndU]
            nn = Nn[LND==lndU]
            v0 = V0[LND==lndU]
            v1 = V1[LND==lndU]
            v2 = V2[LND==lndU]
            t0 = T0[LND==lndU]
            t1 = T1[LND==lndU]
            t2 = T2[LND==lndU]
            [p0,p1,p2,p3] = calcCTFleischnerProb(nd,nn,v0,v1,v2,t0,t1,t2,verb=False)
            
            vclass = np.ndarray.flatten(np.argmax([v0,v1,v2], axis=0))
            tclass = np.ndarray.flatten(np.argmax([t0,t1,t2], axis=0))
        elif 'Volume' in header and 'Text' in header:
            # if volume and texture are given compute Fleischner class
            nd = Nd[LND==lndU]
            v = V[LND==lndU]
            t = T[LND==lndU]
            vclass = calcNodVolClass(v)
            tclass = calcNodTexClass(t)
            
        elif 'VolumeClass' in header and 'TextClass' in header:
            # if volume and texture class are given compute Fleischner class
            nd = Nd[LND==lndU]
            vclass = Vclass[LND==lndU]
            tclass = Tclass[LND==lndU]
            
        f = calcCTFleischnerScore(nd,vclass,tclass,nodprob=.5)
        
        print('LNDb {}'.format(lndU))
        print('Fleischner class: {}'.format(f))
        print('Fleischner class probabilities: {} {} {} {}'.format(p0,p1,p2,p3))
        # return csv with Fleischner score for each CT
        fleischner.append([lndU,f,p0,p1,p2,p3])
    return fleischner

def calcNodVolClass(nodvol):
    # Compute volume class from nodule volume nodvol
    volthr = [100,250]
    if isinstance(nodvol,float) or isinstance(nodvol,int):
        if nodvol>=volthr[0] and nodvol<volthr[1]:
            vclass = 1
        elif nodvol>=volthr[1]:
            vclass = 2
        else:
            vclass = 0
    else: #numpy array
        vclass = np.zeros(nodvol.shape)
        vclass[np.bitwise_and(nodvol>=volthr[0],nodvol<volthr[1])]=1
        vclass[nodvol>=volthr[1]]=2
    return vclass

def calcNodTexClass(nodtex):
    # Compute texture class from texture rating nodtex
    texthr = [7/3,11/3]
    if isinstance(nodtex,float) or isinstance(nodtex,int):
        if nodtex>=texthr[0] and nodtex<=texthr[1]:
            vclass = 1
        elif nodtex>texthr[1]:
            vclass = 2
        else:
            vclass = 0
    else: #numpy array
        vclass = np.zeros(nodtex.shape)
        vclass[np.bitwise_and(nodtex>=texthr[0],nodtex<texthr[1])]=1
        vclass[nodtex>=texthr[1]]=2
    return vclass

if __name__ == "__main__":
    # Compute Fleischner score for all trainset nodules
    fname_gtNodulesFleischner = '/media/root/老王/@data_LNDb/LNDb dataset/trainset_csv/trainNodules.csv'
    gtNodules = readCsv(fname_gtNodulesFleischner)
    gtNodules = joinNodules(gtNodules)
    pdFleischner = calcFleischner(gtNodules)
    len(pdFleischner)

    #
    # # Compute Fleischner score for all predicted nodules (given volume and texture rating/class/probabilities)
    # fname_pdNodulesFleischner = 'predictedNodules.csv'
    # pdNodules = readCsv('input/'+fname_pdNodulesFleischner)
    # pdFleischner = calcFleischner(pdNodules)
    #
    #










