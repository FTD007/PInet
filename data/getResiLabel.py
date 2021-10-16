from Bio.PDB import *

resis = dict(ALA=['N', 'H', 'CA', 'CB', 'C', 'O'],
                 ARG=['N', 'H', 'CA', 'CB', 'CG', 'CD', 'NE', 'HE', 'CZ', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'HH22', 'C', 'O'],
                 ARN=['N', 'H', 'CA', 'CB', 'CG', 'CD', 'NE', 'HE', 'CZ', 'NH1', 'HH11', 'HH12', 'NH2', 'HH21', 'C', 'O'],
                 ASN=['N', 'H', 'CA', 'CB', 'CG', 'OD1', 'ND2', 'HD21', 'HD22', 'C', 'O'],
                 ASP=['N', 'H', 'CA', 'CB', 'CG', 'OD1', 'OD2', 'C', 'O'],
                 ASH=['N', 'H', 'CA', 'CB', 'CG', 'OD1', 'OD2', 'HD2', 'C', 'O'],
                 CYS=['N', 'H', 'CA', 'CB', 'SG', 'C', 'O'],
                 GLN=['N', 'H', 'CA', 'CB', 'CG', 'CD', 'OE1', 'NE2', 'HE21', 'HE22', 'C', 'O'],
                 GLU=['N', 'H', 'CA', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'C', 'O'],
                 GLH=['N', 'H', 'CA', 'CB', 'CG', 'CD', 'OE1', 'OE2', 'HE2', 'C', 'O'],
                 GLY=['N', 'H', 'CA', 'C', 'O'],
                 HIS=['N', 'H', 'CA', 'CB', 'CG', 'ND1', 'HD1', 'CD2', 'NE2', 'CE1', 'C', 'O'],
                 HID=['N', 'H', 'CA', 'CB', 'CG', 'ND1', 'HD1', 'CD2', 'NE2', 'CE1', 'C', 'O'],
                 HIP=['N', 'H', 'CA', 'CB', 'CG', 'CD2', 'ND1', 'HD1', 'CE1', 'NE2', 'HE2', 'C', 'O'],
                 HIE=['N', 'H', 'CA', 'CB', 'CG', 'ND1', 'CE1', 'CD2', 'NE2', 'HE2', 'C', 'O'],
                 ILE=['N', 'H', 'CA', 'CB', 'CG2', 'CG1', 'CD', 'C', 'O'],
                 LEU=['N', 'H', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'C', 'O'],
                 LYS=['N', 'H', 'CA', 'CB', 'CG', 'CD', 'CE', 'NZ', 'HZ1', 'HZ2', 'HZ3', 'C', 'O'],
                 LYN=['N', 'H', 'CA', 'CB', 'CG', 'CD', 'CE', 'NZ', 'HZ1', 'HZ2', 'C', 'O'],
                 MET=['N', 'H', 'CA', 'CB', 'CG', 'SD', 'CE', 'C', 'O'],
                 MSE=['N', 'H', 'CA', 'CB', 'CG', 'SE', 'CE', 'C', 'O'],
                 PHE=['N', 'H', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'C', 'O'],
                 PRO=['N', 'CD', 'CA', 'CB', 'CG', 'C', 'O'],
                 SER=['N', 'H', 'CA', 'CB', 'OG', 'HG', 'C', 'O'],
                 THR=['N', 'H', 'CA', 'CB', 'OG1', 'HG1', 'CG2', 'C', 'O'],
                 TRP=['N', 'H', 'CA', 'CB', 'CG', 'CD2', 'CE2', 'CE3', 'CD1', 'NE1', 'HE1', 'CZ2', 'CZ3', 'CH2', 'C', 'O'],
                 TYR=['N', 'H', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'HH', 'C', 'O'],
                 VAL=['N', 'H', 'CA', 'CB', 'CG1', 'CG2', 'C', 'O']
    )
# hfile=open('/Users/bowendai/Documents/ProteinEng/hydro.csv','r')
# hfile=open('/dartfs-hpc/rc/lab/C/CBKlab/Bdai/pythontools/hydro.csv','r')
hfile=open('hydro.csv','r')
hdic={}
for line in hfile:
    ll=line.split(':')
    if len(ll)>1:

        hdic[ll[0].upper()]=float(ll[1])
        
residuetype=list(hdic.keys())
# print hdic
# print len(hdic)
edic={}
charge=['LYS','ARG','HIS']
necharge=['ASP','GLU']
for aa in hdic:
    if aa in charge:
        edic[aa]=1
    elif aa in necharge:
        edic[aa]=-1
    else:
        edic[aa]=0

def getcontactbyabag(folder,file,ab='',ag=''):
    if ab=='':
        i1=file.find('-')
        i2=file.find('-',i1+1)
        i3=min(file.find('.pdb'),file.find('-',i2+1))
        ab=file[i1+1:i2]
        ag=file[i2+1:i3]

    atomset=['CA','CB']
    chain=ab
    parser=PDBParser()
    # structure = parser.get_structure('C', '3ogo-bg.pdb')
    structure = parser.get_structure('C', folder+file)
    allchain=[]
    for c in ab:
        allchain.append(c)
    for c in ag:
        allchain.append(c)
    newdick={}
    newdick['r']=[]
    newdick['l']=[]
    labeldick={}
    labeldick['r'] = [[],[]]
    labeldick['l'] = [[],[]]

    for chain in allchain:
        try:
            test=structure[0][chain]
        except:
            continue
        for resi in structure[0][chain]:
            if resi.get_resname() not in hdic.keys():
                continue
            # print resi.get_resname()
            cen=[0,0,0]
            count=0
            for atom in resi:
                # print atom.get_coord()
                # print list(atom.get_vector())
                cen[0]+=atom.get_coord()[0]
                cen[1] += atom.get_coord()[1]
                cen[2] += atom.get_coord()[2]
                count+=1
            cen=[coor*1.0/count for coor in cen]
            if chain in ag:
                newdick['r'].append(cen)
                labeldick['r'][0].append(hdic[resi.get_resname()])
                labeldick['r'][1].append(edic[resi.get_resname()])
            else:
                newdick['l'].append(cen)
                labeldick['l'][0].append(hdic[resi.get_resname()])
                labeldick['l'][1].append(edic[resi.get_resname()])
    return newdick,labeldick,ab,ag

def gethydro(file):

    atomset=['CA','CB']
    parser=PDBParser()
    structure = parser.get_structure('C', file)
    newdick=[]
    labeldick= [[],[]]

    for chain in structure[0]:
        for resi in chain:
            if resi.get_resname() not in hdic.keys():
                continue
#             print resi.get_resname()
            cen=[0,0,0]
            count=0
            for atom in resi:
                # print atom.get_coord()
                # print list(atom.get_vector())
                cen[0]+=atom.get_coord()[0]
                cen[1] += atom.get_coord()[1]
                cen[2] += atom.get_coord()[2]
                count+=1
            cen=[coor*1.0/count for coor in cen]
            newdick.append(cen)
            labeldick[0].append(hdic[resi.get_resname()])
            labeldick[1].append(edic[resi.get_resname()])
            
    return newdick,labeldick

def getlabels(file):
#     print(residuetype)
    atomset=['CA','CB']
    parser=PDBParser()
    structure = parser.get_structure('C', file)
    newdick=[]
    labeldick= [[],[],[],[]]

    for chain in structure[0]:
        for resi in chain:
            if resi.get_resname() not in hdic.keys():
                continue
#             print resi.get_resname()
            cen=[0,0,0]
            count=0
            for atom in resi:
                # print atom.get_coord()
                # print list(atom.get_vector())
                cen[0] = atom.get_coord()[0]
                cen[1] = atom.get_coord()[1]
                cen[2] = atom.get_coord()[2]
                newdick.append(cen)
                labeldick[0].append(hdic[resi.get_resname()])
                labeldick[1].append(edic[resi.get_resname()])
                labeldick[2].append(residuetype.index(resi.get_resname()))
#                 print(resi.get_resname())
                if atom.get_name() in resis[resi.get_resname()]:
                    labeldick[3].append(resis[resi.get_resname()].index(atom.get_name()))
                else:
                    labeldick[3].append(-1)
    return newdick,labeldick
