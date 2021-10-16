from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

# standard dataset
class ShapeNetDataset3(data.Dataset):
    def __init__(self,
                 root,
                 npoints=3000,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True,
                 indim=5,
                 rs=0):

        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        self.indim=indim
        self.rs=rs

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split',
                                 'shuffled_{}_file_list_{}.json'.format(split, class_choice[0][0]))
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label',
                                                                      uuid + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        if self.rs==1:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
            point_set = point_set[choice, :]
            seg = seg[choice]

        # noemalize flag
        norm=1
        cnorm=0

        coordset = point_set[:, 0:3]
        featset = point_set[:, 3:]
        if norm==1:
            if self.indim==4:
                # electrostatic normalize
                eset = (np.expand_dims(point_set[:, 3], axis=1)) / np.sqrt(np.max(point_set[:, 3] ** 2))
                # hydrophobicity nomalize
                hset = (np.expand_dims(point_set[:, 4], axis=1)) / np.sqrt(np.max(point_set[:, 4] ** 2))
                featset=eset
            elif self.indim>4 and self.indim<=6:
                featset=point_set[:,3:self.indim]
                featset=featset/np.sqrt(np.max(featset ** 2,axis=0))
            else:
                featset1 = point_set[:, 3:5]
                featset2= point_set[:,5:self.indim]
                featset1 = featset1 / np.sqrt(np.max(featset1 ** 2, axis=0))
                featset=np.concatenate((featset1,featset2),axis=1)
        coordset = coordset - np.expand_dims(np.mean(coordset, axis=0), 0)  # center
        # size normalize
        if cnorm == 1:
            dist = np.max(np.sqrt(np.sum(coordset ** 2, axis=1)), 0)
            coordset = coordset / dist  # scale
        if self.indim > 3:
            point_set[:, 0:self.indim] = np.concatenate((coordset, featset), axis=1)
        else:
            point_set=coordset
        point_set = point_set[:, 0:self.indim]
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
        else:
            theta = 0

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)




class ShapeNetDataset3aug(data.Dataset):
    def __init__(self,
                 root,
                 npoints=3000,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True,
                 indim=5,
                 rs=0,
                 fold=''):

        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        self.indim = indim
        self.rs = rs
        self.fold=fold

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        if fold=='':
            splitfile = os.path.join(self.root, 'train_test_split',
                                 'shuffled_{}_file_list_{}.json'.format(split, class_choice[0][0]))
        else:
            splitfile = os.path.join(self.root, 'tts'+fold,
                                     'shuffled_{}_file_list_{}.json'.format(split, class_choice[0][0]))
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                if (len(uuid)==6):
                    self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label',
                                                                      uuid + '.seg')))
                elif (len(uuid)==7)&(int(uuid[-1])>5):
                    continue
                else:
                    self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                             os.path.join(self.root, category, 'points_label',
                                                                          uuid + '.seg')))
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        if self.rs == 1:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            # resample
            point_set = point_set[choice, :]
            seg = seg[choice]
        norm = 1
        cnorm = 0

        coordset = point_set[:, 0:3]
        featset = point_set[:, 3:]
        if norm == 1:
            if self.indim == 4:
                eset = (np.expand_dims(point_set[:, 3], axis=1)) / np.sqrt(np.max(point_set[:, 3] ** 2))
                hset = (np.expand_dims(point_set[:, 4], axis=1)) / np.sqrt(np.max(point_set[:, 4] ** 2))
                featset = eset
            elif self.indim > 4 and self.indim <= 6:

                featset = point_set[:, 3:self.indim]
                featset = featset / np.sqrt(np.max(featset ** 2, axis=0))
            else:
                featset1 = point_set[:, 3:5]
                featset2 = point_set[:, 5:self.indim]
                featset1 = featset1 / np.sqrt(np.max(featset1 ** 2, axis=0))
                featset = np.concatenate((featset1, featset2), axis=1)
        coordset = coordset - np.expand_dims(np.mean(coordset, axis=0), 0)  # center
        if cnorm == 1:
            dist = np.max(np.sqrt(np.sum(coordset ** 2, axis=1)), 0)
            coordset = coordset / dist  # scale
        if self.indim > 3:
            point_set[:, 0:self.indim] = np.concatenate((coordset, featset), axis=1)

        else:
            point_set = coordset
        point_set = point_set[:, 0:self.indim]
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
        else:
            theta = 0

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)



if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

