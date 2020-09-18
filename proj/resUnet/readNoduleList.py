import numpy as np
import math

from utils import readCsv, writeCsv

def nodEqDiam(vol):
    # Calc nodule equivalent diameter from volume vol
    return 2*(vol*3/(4*math.pi))**(1/3)

def joinNodules(nodules,verb=False):
    # join nodules from different radiologists (if within radiusor 3mm from each other)
    header = nodules[0]
    lines = nodules[1:]
    lndind = header.index('LNDbID')
    radind = header.index('RadID')
    fndind = header.index('FindingID')
    xind = header.index('x')
    yind = header.index('y')
    zind = header.index('z')
    nodind = header.index('Nodule')
    volind = header.index('Volume')
    texind = header.index('Text')
    LND = [int(line[lndind]) for line in lines]
    
    # Match nodules
    nodules = [['LNDbID','RadID','RadFindingID','FindingID','x','y','z','AgrLevel','Nodule','Volume','Text']]
    for lndU in np.unique(LND): #within each CT
        nodlnd = [line for lnd,line in zip(LND,lines) if lnd==lndU]
        dlnd = [nodEqDiam(float(n[volind])) for n in nodlnd]
        nodnew = [] #create merged nodule list
        dnew = []
        for iself,n in enumerate(nodlnd): #for each nodule
            dself = dlnd[iself]/2
            rself = n[radind]
            nodself = n[nodind]
            match = False
            for inew,nnew in enumerate(nodnew): #check distance with every nodule in merged list
                if not float(rself) in nnew[radind] and float(nodself) in nnew[nodind]:
                    dother = max(dnew[inew])/2
                    dist = ((float(n[xind])-np.mean(nnew[xind]))**2+
                            (float(n[yind])-np.mean(nnew[yind]))**2+
                            (float(n[zind])-np.mean(nnew[zind]))**2)**.5
                    if dist<max(max(dself,dother),3): # if distance between nodules is smaller than maximum radius or 3mm
                        match = True
                        for f in range(len(n)):
                            nnew[f].append(float(n[f])) #append to existing nodule in merged list
                        dnew[inew].append(np.mean([nodEqDiam(v) for v in nodnew[inew][volind]]))
                        break
            if not match:
                nodnew.append([[float(l)] for l in n]) #otherwise append new nodule to merged list
                dnew.append([np.mean([nodEqDiam(v) for v in nodnew[-1][volind]])])
            
            if verb:
                print(iself)
                for inew,nnew in enumerate(nodnew):
                    print(nnew,dnew[inew])
        
        # Merge matched nodules
        for ind,n in enumerate(nodnew):
            agrlvl = n[nodind].count(1.0)
            if agrlvl>0: #nodules
                nod = 1
                vol = np.mean(n[volind]) #volume is the average of all radiologists
                tex = np.mean(n[texind]) #texture is the average of all radiologists
            else: #non-nodules
                nod = 0
                vol = 4*math.pi*1.5**3/3 #volume is the minimum for equivalent radius 3mm
                tex = 0
            nodules.append([int(n[lndind][0]),
                            ','.join([str(int(r)) for r in n[radind]]), #list radiologist IDs
                            ','.join([str(int(f)) for f in n[fndind]]), #list radiologist finding's IDs
                            ind+1, # new finding ID
                            np.mean(n[xind]), #centroid is the average of centroids
                            np.mean(n[yind]),
                            np.mean(n[zind]),
                            agrlvl, # number of radiologists that annotated the finding (0 if non-nodule)
                            nod,
                            vol,
                            tex])
    if verb:
        for n in nodules:
            print(n)
    return nodules

if __name__ == "__main__":
    # Merge nodules from train set
    prefix = 'train'
    fname_gtNodulesFleischner = '{}Nodules.csv'.format(prefix)
    gtNodules = readCsv(fname_gtNodulesFleischner)
    for line in gtNodules:
        print(line)
    gtNodules = joinNodules(gtNodules)
    writeCsv('{}Nodules_gt.csv'.format(prefix),gtNodules) #write to csv