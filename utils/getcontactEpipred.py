from Bio.PDB import *
import numpy as np
from Bio import pairwise2
def getcontactbyabag(folder,file,d=9,ab='',ag=''):
    # folder='/Users/bowendai/Documents/ProteinEng/'
    # file='3w9e-C-AB.pdb'
    if ab=='':
        i1=file.find('-')
        i2=file.find('-',i1+1)
        i3=min(file.find('.pdb'),file.find('-',i2+1))
        ab=file[i1+1:i2]
        ag=file[i2+1:i3]
    # print ab,ag
    # d=5
    atomset=['CA','CB']
    chain=ab
    parser=PDBParser()
    # structure = parser.get_structure('C', '3ogo-bg.pdb')
    structure = parser.get_structure('C', folder+file)
    atom_list = Selection.unfold_entities(structure, 'A')  # A for atoms
    ns = NeighborSearch(atom_list)
    pairs = ns.search_all(d)
    s=0
    # pairdic=defaultdict(set)
    # pairdic={}
    # count = 0
    # co=[0,0,0]
    # pc=0
    # dick={ag:set([])}
    dick={}
    for aabb in ab:
        dick[aabb]=set()
    for aagg in ag:
        dick[aagg]=set()
    for p in pairs:
        # print p[0]
        # if not (str(p[0])[6:8] in atomset and str(p[1])[6:8] in atomset):
        #     continue
        if str(p[0])[6]=='H' or str(p[1])[6]=='H':
            continue
        # print p
        try:
            residue_list = Selection.unfold_entities(p, 'R')  # R for residues
            chain_list = Selection.unfold_entities(p, 'C')  # C for chains
        except:
            print p
            continue
        if len(chain_list)!=2:
            continue
        # print Selection.unfold_entities(p[0], 'R')[0]
        # print 1
        # print p[1]
        # print Selection.unfold_entities(p, 'R')
        # print p
        # try:
        #     Selection.unfold_entities(p, 'R')
        # except:
        #     continue
        if p[0].is_disordered():
            continue
        if p[1].is_disordered():
            continue
        residue_list=[Selection.unfold_entities(p[0], 'R')[0],Selection.unfold_entities(p[1], 'R')[0]]
        chain_list = [Selection.unfold_entities(p[0], 'C')[0], Selection.unfold_entities(p[1], 'C')[0]]
        # print chain_list
        cc=[c._id for c in chain_list]
        # print set(cc)
        # print set(ag)
        # print set(cc)&set(ag)
        if len(set(cc)&set(ag))==0:
            continue
        if len(set(cc)&set(ab))==0:
            continue
        # print [pi._id for pi in chain_list]
        # print [pi._id[1] for pi in residue_list]
        # print residue_list[0].get_resname()
        # dick[chain_list[0]._id].add((residue_list[0]._id[1],residue_list[0].get_resname()))
        # dick[chain_list[1]._id].add((residue_list[1]._id[1],residue_list[1].get_resname()))
        dick[chain_list[0]._id].add(residue_list[0]._id[1])
        dick[chain_list[1]._id].add(residue_list[1]._id[1])

    newdick={}
    newdick['r']=[]
    newdick['l']=[]
    labeldick={}
    labeldick['r'] = []
    labeldick['l'] = []
    residick = {}
    residick['r'] = []
    residick['l'] = []
    # for nc in dick:
    #     newdick[nc]=set()
    #     for nresi in dick[nc]:
    #         try:
    #             c=[0,0,0]
    #             for natom in structure[0][nc][nresi[0]]:
    #                 try:
    #                     print natom.get_coord()
    #
    #                 except:
    #                     print 1
    #         except:
    #             print nc,nresi[0]
    for chain in dick:
        for resi in structure[0][chain]:
            cen=[0,0,0]
            count=0
            for atom in resi:
                # if 'H' in atom.get_name():
                #     continue
                # print atom.get_coord()
                # print list(atom.get_vector())
                cen[0]+=atom.get_coord()[0]
                cen[1] += atom.get_coord()[1]
                cen[2] += atom.get_coord()[2]
                count+=1
            cen=[coor*1.0/count for coor in cen]
            if chain in ag:
                newdick['r'].append(cen)
                residick['r'].append(resi._id[1])
                if resi._id[1] in dick[chain]:
                    labeldick['r'].append(1)
                else:
                    labeldick['r'].append(0)
            else:
                newdick['l'].append(cen)
                residick['l'].append(resi._id[1])
                if resi._id[1] in dick[chain]:
                    labeldick['l'].append(1)
                else:
                    labeldick['l'].append(0)
    # print newdick
    # print labeldick
    # for key in newdick:
    #     print key
    #     print len(newdick[key])
    #     print len(labeldick[key])
    # return dick,ag,ab
    # return newdick
    return newdick,labeldick,residick,ab,ag

def getsppider(file):
    parser = PDBParser()
    # structure = parser.get_structure('C', '3ogo-bg.pdb')
    structure = parser.get_structure('C', file)
    residic=[]
    bdic=[]
    for c in structure[0]:
        for r in c:
            residic.append(r._id[1])
            for a in r:
                if a.get_bfactor()>0:
                    bdic.append(1)
                else:
                    bdic.append(0)
                break
    # print len(residic)
    # print len(bdic)
    return residic,bdic

def getsppider2(file):
    parser = PDBParser()
    # structure = parser.get_structure('C', '3ogo-bg.pdb')
    structure = parser.get_structure('C', file)
    residic = []
    # tempo=[]
    newdic =[]
    labeldic= []
    bdic = []
    cd=[]
    mark=0
    for c in structure[0]:
        for resi in c:
            # residic.append(resi._id[1])
            cen = [0, 0, 0]
            count = 0
            for atom in resi:
                # print atom.get_coord()
                # print list(atom.get_vector())
                # if 'H' in atom.get_name():
                #     continue
                cen[0] += atom.get_coord()[0]
                cen[1] += atom.get_coord()[1]
                cen[2] += atom.get_coord()[2]
                count += 1

                # residic.append(resi._id[1])
                residic.append(mark)
                newdic.append([atom.get_coord()[0],atom.get_coord()[1],atom.get_coord()[2]])
            cen = [coor * 1.0 / count for coor in cen]
            mark+=1
            cd.append(cen)
            # labeldic.append(1)

    # print len(residic)
    # print len(bdic)
    return residic, np.asarray(newdic),cd

def getalign(f1,f2):
    parser = PDBParser()
    # structure = parser.get_structure('C', '3ogo-bg.pdb')
    s1 = parser.get_structure('C1', f1)
    s2 = parser.get_structure('C2', f2)

    ppb = PPBuilder()

    l1 = ''
    l2 = ''
    for pp in ppb.build_peptides(s1):
        l1 += pp.get_sequence()

    for pp in ppb.build_peptides(s2):
        l2 += pp.get_sequence()

    align = pairwise2.align.globalxx(l1, l2)
    return align
#
# #
# # folder='/Users/bowendai/Documents/ProteinEng/'
# # folder='/Users/bowendai/Documents/Topoprotein/germ3/selectpdb/'
# folder='/Users/bowendai/Documents/nn/pointprotein/structure-clean2/'
# file='4jr9.pdb'
# folder='/Users/bowendai/Documents/nn/pointprotein/visual/'
# file='1dqj.pdb'
# # # file='clean1g8m-HL-G.pdb'
# # # dick,ag,ab=getcontactbyabag(folder,file)
# nd,ld,rd,ag,ab=getcontactbyabag(folder,file,d=4.5,ab='AB',ag='C')
# print nd
# print ld
# print rd
# for i,label in enumerate(ld['r']):
#     if label==1:
#         print rd['r'][i]
# print dick
# print ag
# print ab
# for l in dick:
#     print l
#     print len(dick[l])
#     print sorted(dick[l])
# #
# # for i in structure[0][chain]:
# #     print i
# getsppider2('/Users/bowendai/Documents/nn/pointprotein/sppider/4ene_B.txt')

# f1='/Users/bowendai/Documents/nn/pointprotein/structure-clean-fixed-u/1HE1-l.pdb'
# f2='/Users/bowendai/Documents/Topoprotein/sabdab/benchmark5/structures-fixed/1HE1_l.pdb'
# print getalign(f1,f2)