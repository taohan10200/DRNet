##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/Sha-Lab/FEAT
## Tianjin University
## liuyaoyao@tju.edu.cn
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Sampler for dataloader. """
import torch
import numpy as np
import  random
class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, labels, frame_intervals, n_per):
        self.frame_intervals = frame_intervals
        self.n_sample = len(labels)
        self.n_batch =   self.n_sample// n_per
        self.n_per = n_per
        self.scenes = []
        self.scene_id = {}
        for idx, label in enumerate(labels):
            scene_name = label['scene_name']
            if scene_name not in self.scene_id.keys():
                self.scene_id.update({scene_name:0})
            self.scene_id[scene_name]+=1
            self.scenes.append(scene_name)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            frame_a = torch.randperm(self.n_sample )[:self.n_per]
            for c in frame_a:
                scene_name = self.scenes[c]
                # print(c)
                tmp_intervals = random.randint(self.frame_intervals[0],
                                               min(self.scene_id[scene_name]//2,self.frame_intervals[1]))
                if c<self.n_sample-tmp_intervals:
                    if self.scenes[c + tmp_intervals] == scene_name:
                        pair_c = c + tmp_intervals
                    else:
                        pair_c = c
                        c = c- tmp_intervals
                else:
                    pair_c = c
                    c = c - tmp_intervals
                assert self.scenes[c] == self.scenes[pair_c]
                batch.append(torch.tensor([c, pair_c]))

            batch = torch.stack(batch).reshape(-1)
            yield batch


class Val_CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, labels, frame_intervals, n_per):
        self.frame_intervals = frame_intervals

        self.n_sample = len(labels)
        self.n_batch =   self.n_sample // n_per   #there is no need to evaluate all frames
        self.n_per = n_per
        self.scenes = []
        scene_id = {}
        for idx, label in enumerate(labels):
            scene_name = label['scene_name']
            if scene_name not in scene_id.keys():
                scene_id.update({scene_name:[]})
            scene_id[scene_name].append(idx)
            self.scenes.append(scene_name)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            frame_a = torch.randperm(self.n_sample )[:self.n_per]
            for c in frame_a:
                scene_name = self.scenes[c]
                # print(c)
                if c<self.n_sample-self.frame_intervals:
                    if self.scenes[c + self.frame_intervals] == scene_name:
                        pair_c = c + self.frame_intervals
                    else:
                        pair_c = c
                        c = c- self.frame_intervals
                else:
                    pair_c = c
                    c = c - self.frame_intervals
                assert self.scenes[c] == self.scenes[pair_c]
                batch.append(torch.tensor([c, pair_c]))

            batch = torch.stack(batch).t().reshape(-1)

            yield batch
