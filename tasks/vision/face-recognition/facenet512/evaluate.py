#!/usr/bin/env python3
# Copyright 2022-2025 NXP
# SPDX-License-Identifier: MIT
import argparse
import sys
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
from sklearn.decomposition import PCA
import pickle
import os
import io
from tqdm import tqdm
from skimage.io import imread
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)
    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]
        
class eval_callback(tf.keras.callbacks.Callback):
    def __init__(self,model_path, test_bin_file, batch_size=1, eval_freq=1, flip=True, PCA_acc=False):
        super(eval_callback, self).__init__()
        #Reading bins and labels
        bins, issame_list = np.load(test_bin_file, encoding="bytes", allow_pickle=True)
        ds= tf.data.Dataset.from_tensor_slices(bins)
        #Normalization: inputs should be normalized 
        _imread = lambda xx: (tf.cast(tf.image.decode_image(xx, channels=3),"float32")/255.0)
        ds= ds.map(_imread)
        self.model_path=model_path
        self.ds=ds.batch(batch_size)
        self.test_issame = np.array(issame_list).astype("bool")
        self.test_names = os.path.splitext(os.path.basename(test_bin_file))[0]
        self.steps = int(np.ceil(len(bins) / batch_size))
        self.max_accuracy, self.cur_acc, self.acc_thresh = 0.0, 0.0, 0.0
        self.eval_freq, self.flip, self.PCA_acc =  eval_freq, flip, PCA_acc
        self.on_epoch_end = lambda epoch=0, logs=None: self.__eval_func__(epoch, logs, eval_freq=1)
    def __do_predict__(self):
        embs = []
        embs_f=[]
        embeddings_list=[]
        interpreter = tf.lite.Interpreter(self.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_scale, input_zero_point = input_details[0]["quantization"]  

        print("Model_loaded")
        for img_batch in tqdm(self.ds, "Evaluating " + self.test_names, total=self.steps):
            #Calculating non-flipped image embeddings
            interpreter.set_tensor(input_details[0]['index'], tf.cast(tf.image.resize(img_batch[...,::-1],[160,160])/input_scale+input_zero_point,tf.uint8))
            interpreter.invoke()
            emb = interpreter.get_tensor(output_details[0]['index'])
            #Calculating flipped image embeddings        
            if self.flip:  
                interpreter.set_tensor(input_details[0]['index'], tf.cast(tf.image.flip_left_right(tf.image.resize(img_batch[...,::-1],[160,160]))/input_scale+input_zero_point,tf.uint8))
                interpreter.invoke()
                emb_f = interpreter.get_tensor(output_details[0]['index']) 
            embs.append(emb)
            embs_f.append(emb_f)               
        embeddings_list.append(np.squeeze(np.array(embs)))
        embeddings_list.append(np.squeeze(np.array(embs_f)))                        
        return embeddings_list  


    def __eval_func__(self, cur_step=0, logs=None, eval_freq=1): 
        
        if cur_step % eval_freq != 0:
            return
        else:
            cur_step = str(cur_step + 1)
        dists=[]
        embeddings_list = self.__do_predict__()
        if not np.alltrue(np.isfinite(embeddings_list)):
            tf.print("NAN in embs")
            return
        embeddings = embeddings_list[0].copy()
        embeddings = sklearn.preprocessing.normalize(embeddings)
        acc1 = 0.0
        std1 = 0.0
        tpr, fpr, accuracy, val, val_std, far,thresh1,thresh2,acc_baseline,thresh_baseline  = evaluate(embeddings, self.test_issame, nrof_folds=10)
        acc1, std1 = np.mean(accuracy), np.std(accuracy)
        print('FAR target',1e-3)
        print("---------------No flip----------------")
        print('Baseline accuracy:',acc_baseline)
        print('Baseline threshold:',thresh_baseline)
        print('FAR acheived:', far)
        print('Accuracy:',acc1,' +-',std1)
        print("Threshold", thresh1)
        print('TAR@FAR :',val,' +-',val_std)
        print("FAR based threshold", thresh2)
        embeddings = embeddings_list[0] + embeddings_list[1]
        embeddings = sklearn.preprocessing.normalize(embeddings)
        tpr, fpr, accuracy, val, val_std, far,thresh1,thresh2,acc_baseline,thresh_baseline = evaluate(embeddings, self.test_issame, nrof_folds=10)
        acc2, std2 = np.mean(accuracy), np.std(accuracy)
        print("---------------With Flip----------------")
        print('Baseline accuracy:',acc_baseline)
        print('Baseline threshold:',thresh_baseline)
        print('FAR acheived:', far)
        print('Accuracy:',acc2,' +-',std2)
        print("Threshold", thresh1)
        print('TAR@FAR :',val,' +-',val_std)
        print("FAR based threshold", thresh2)
def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
            def calculate_roc(thresholds,
                        embeddings1,
                        embeddings2,
                        actual_issame,
                        nrof_folds=10,
                        pca=0):
                def calculate_accuracy(threshold, dist, actual_issame):
                    predict_issame = np.less(dist, threshold)
                    tp = np.sum(np.logical_and(predict_issame, actual_issame))
                    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
                    tn = np.sum(
                        np.logical_and(np.logical_not(predict_issame),
                                        np.logical_not(actual_issame)))
                    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

                    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
                    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
                    acc = float(tp + tn) / dist.size
                    return tpr, fpr, acc   
                
                assert (embeddings1.shape[0] == embeddings2.shape[0])
                assert (embeddings1.shape[1] == embeddings2.shape[1])
                nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
                nrof_thresholds = len(thresholds)
                k_fold = LFold(n_splits=nrof_folds, shuffle=False)

                tprs = np.zeros((nrof_folds, nrof_thresholds))
                fprs = np.zeros((nrof_folds, nrof_thresholds))
                accuracy = np.zeros((nrof_folds))
                indices = np.arange(nrof_pairs)


                if pca == 0:
                    diff = np.subtract(embeddings1, embeddings2)
                    dist = np.sum(np.square(diff), 1)
                    scaler = MinMaxScaler((0, 1))
                    dist = scaler.fit_transform(dist.reshape(-1, 1)).flatten()
                for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
                    if pca > 0:
                        print('doing pca on', fold_idx)
                        embed1_train = embeddings1[train_set]
                        embed2_train = embeddings2[train_set]
                        _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
                        pca_model = PCA(n_components=pca)
                        pca_model.fit(_embed_train)
                        embed1 = pca_model.transform(embeddings1)
                        embed2 = pca_model.transform(embeddings2)
                        embed1 = sklearn.preprocessing.normalize(embed1)
                        embed2 = sklearn.preprocessing.normalize(embed2)
                        diff = np.subtract(embed1, embed2)
                        dist = np.sum(np.square(diff), 1)
                        scaler = MinMaxScaler((0, 1))
                        dist = scaler.fit_transform(dist.reshape(-1, 1)).flatten()
                    acc_train = np.zeros((nrof_thresholds))
                    for threshold_idx, threshold in enumerate(thresholds):
                        _, _, acc_train[threshold_idx] = calculate_accuracy(
                            threshold, dist[train_set], actual_issame[train_set])
                    best_threshold_index = np.argmax(acc_train)
                    for threshold_idx, threshold in enumerate(thresholds):
                        tprs[fold_idx,
                            threshold_idx], fprs[fold_idx,
                                                threshold_idx], _ = calculate_accuracy(
                                                    threshold, dist[test_set],
                                                    actual_issame[test_set])
                    _, _, accuracy[fold_idx] = calculate_accuracy(
                        thresholds[best_threshold_index], dist[test_set],
                        actual_issame[test_set])

                tpr = np.mean(tprs, 0)
                fpr = np.mean(fprs, 0)
                return tpr, fpr, accuracy,thresholds[best_threshold_index]
            def calculate_val(thresholds,
                    embeddings1,
                    embeddings2,
                    actual_issame,
                    far_target,
                    nrof_folds=10):
                def calculate_val_far(threshold, dist, actual_issame):
                    predict_issame = np.less(dist, threshold)
                    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
                    false_accept = np.sum(
                        np.logical_and(predict_issame, np.logical_not(actual_issame)))
                    n_same = np.sum(actual_issame)
                    n_diff = np.sum(np.logical_not(actual_issame))
                    val = 0 if (n_same==0) else float(true_accept) / float(n_same)
                    far = 0 if (n_diff==0) else float(false_accept) / float(n_diff)
                    return val, far
                assert (embeddings1.shape[0] == embeddings2.shape[0])
                assert (embeddings1.shape[1] == embeddings2.shape[1])
                nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
                nrof_thresholds = len(thresholds)
                k_fold = LFold(n_splits=nrof_folds, shuffle=False)

                val = np.zeros(nrof_folds)
                far = np.zeros(nrof_folds)

                diff = np.subtract(embeddings1, embeddings2)
                dist = np.sum(np.square(diff), 1)
                scaler = MinMaxScaler((0, 1))
                dist = scaler.fit_transform(dist.reshape(-1, 1)).flatten()
                indices = np.arange(nrof_pairs)

                for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

                    # Find the threshold that gives FAR = far_target
                    far_train = np.zeros(nrof_thresholds)
                    for threshold_idx, threshold in enumerate(thresholds):
                        _, far_train[threshold_idx] = calculate_val_far(
                            threshold, dist[train_set], actual_issame[train_set])
                    if np.max(far_train) >= far_target:
                        f = interpolate.interp1d(far_train, thresholds, kind='slinear')
                        threshold = f(far_target)
                    else:
                        threshold = 0.0

                    val[fold_idx], far[fold_idx] = calculate_val_far(
                        threshold, dist[test_set], actual_issame[test_set])

                val_mean = np.mean(val)
                far_mean = np.mean(far)
                val_std = np.std(val)
                return val_mean, val_std, far_mean,threshold
            
            thresholds = np.arange(0, 4, 0.01)
            embeddings1 = embeddings[0::2]
            embeddings2 = embeddings[1::2]
            dists=(embeddings1 * embeddings2).sum(1)
            tt = np.sort(dists[actual_issame[: dists.shape[0]]])
            ff = np.sort(dists[np.logical_not(actual_issame[: dists.shape[0]])])
            t_steps = int(0.1 * ff.shape[0])
            acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
            acc_max_indx = np.argmax(acc_count)
            acc_max = acc_count[acc_max_indx] / dists.shape[0]
            acc_thresh = ff[acc_max_indx - t_steps]
            tpr, fpr, accuracy,thresh1 = calculate_roc(thresholds,
                                            embeddings1,
                                            embeddings2,
                                            np.asarray(actual_issame),
                                            nrof_folds=nrof_folds,
                                            pca=pca)
            thresholds = np.arange(0, 4, 0.001)
            #Evaluating the model by fixing the FAR to 1e-3
            val, val_std, far,thresh2 = calculate_val(thresholds,
                                            embeddings1,
                                            embeddings2,
                                            np.asarray(actual_issame),
                                            1e-3,
                                            nrof_folds=nrof_folds)
            return tpr, fpr, accuracy, val, val_std, far,thresh1,thresh2,acc_max,acc_thresh      
