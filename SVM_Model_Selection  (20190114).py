import preprocess as pre
import codecs
import sklearn
import random
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from datetime import datetime
import joblib
############################################################################################
########################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#################################
########################              function             #################################
########################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#################################
############################################################################################
def percentile_entropy(entropy_trans):
    length = len(entropy_trans)
    per_en = {}
    for i,entropy in enumerate(entropy_trans):
        per_en[(i+1)/length] = entropy
    return(per_en)

def dict_transfer(percentile_entropy):
    new_dict = percentile_entropy.copy()
    keys = list(percentile_entropy.keys())
    for i in range(1,len(keys)):
        now = keys[i]
        past = keys[i-1]
        if (percentile_entropy[now]<percentile_entropy[past]):
            new_dict[now] = 1
        else:
            new_dict[now] = 0
    new_dict.pop(keys[0])
    return(new_dict)

###
def lower_float(float_digit, num_digit=1):
    splited_float = str(float_digit).split('.')
    return(float(splited_float[0]+'.'+splited_float[1][:num_digit]))

def no_one_round(float_digit,num):
    temp = round(float_digit+0.05,num)
    if temp > 1:
        return(round(1,num))
    else:
        return(temp)

### gather into percentile
def minus_in_percentile(percen_dict,minus_range=range(1,11)):####adjust minus_range (like hyperpara)
    res = {}
    for i in minus_range:
        res[i/10] = []
    for k in percen_dict.keys():
        res[no_one_round(k,1)].append(percen_dict[k])
        #print(k,no_one_round(k+0.05,1),percen_dict[k])
    #print(res)
    keys = list(res.keys())
    key0 = keys[0]
    #print(key0)
    if len(res[key0]) == 0:
        res[key0] = [0]
    for i in range(1,len(keys)):
        if len(res[keys[i]]) == 0:
            res[keys[i]] = res[keys[i-1]].copy()
    return(res)

######################################################
### calculate probability within percentile dict ### for accumulated trans
def DictTransfer2Prob(minus_in_percentile_dict):
    minus_prob = {}
    length = 0
    accum_1 = 0
    for k in minus_in_percentile_dict.keys():
        length += len(minus_in_percentile_dict[k])
        accum_1 += sum(minus_in_percentile_dict[k])
        minus_prob[k] = accum_1/length
    return(minus_prob)

######################################################
### calculate probability per percentile dict ### for accumulated trans
def DictTransfer2PerProb(minus_in_percentile_dict):
    minus_prob = {}
    for k in minus_in_percentile_dict.keys():
        length = len(minus_in_percentile_dict[k])
        accum_1 = sum(minus_in_percentile_dict[k])
        minus_prob[k] = accum_1/length
    return(minus_prob)

######################################################
### calculate probability within percentile dict ### for accumulated trans
def AverageAccumValue(minus_in_percentile_dict):
    minus_prob = {}
    length = 0
    accum_1 = 0
    for k in minus_in_percentile_dict.keys():
        length += len(minus_in_percentile_dict[k])
        accum_1 += sum(minus_in_percentile_dict[k])
        minus_prob[k] = accum_1/length
    return(minus_prob)

######################################################
### calculate probability per percentile dict ### for ordinary trans
def AveragePerValue(minus_in_percentile_dict):
    minus_prob = {}
    for k in minus_in_percentile_dict.keys():
        length = len(minus_in_percentile_dict[k])
        accum_1 = sum([float(a) for a in minus_in_percentile_dict[k]])
        minus_prob[k] = accum_1 / length
    return (minus_prob)

def EvaluateRes(ytrue,yhat,positive = 1):
    tp,tn,fp,fn = [0,0,0,0]
    for y1now,y2now in zip(ytrue,yhat):
        if y1now == positive and y2now == positive:
            tp += 1
        elif y1now == positive and y2now != positive:
            fn += 1
        elif y1now != positive and y2now == positive:
            fp += 1
        elif y1now != positive and y2now != positive:
            tn += 1
    return(tp,tn,fp,fn)

def SplitData(data, rs):
    train_datas = []
    test_datas = []
    #f = codecs.open(dst+file_name+'.txt','w')
    for train_index,test_index in rs.split(data):
        train_datas.append(data[train_index])
        test_datas.append(data[test_index])
        #f.write('train: ' + ' '.join([str(a) for a in train_index]) + '\n')
        #f.write('val: ' + ' '.join([str(a) for a in test_index]) + '\n\n')
    #f.close()
    return(train_datas,test_datas)
'''
def CreateFoldsDataset(trains_1, trains_2, tests_1, tests_2):
    train_folds = []
    test_folds = []
    for train_1, train_2 in zip(trains_1, trains_2):
        train_folds.append(np.concatenate((train_1, train_2), axis=0))
    for test_1, test_2 in zip(tests_1, tests_2):
        test_folds.append(np.concatenate((test_1, test_2), axis=0))
    return(train_folds,test_folds)
'''

def CreateFoldsDataset(dataset1, dataset2, rs):
    tvt ={'train':{},'val':{},'test':{}}
    tvs1, tests1 = SplitData(dataset1, rs)
    tvs2, tests2 = SplitData(dataset2, rs)
    for i, test_pair in enumerate(zip(tests1, tests2)):
        tvt['test'][i] = np.concatenate((test_pair[0], test_pair[1]), axis=0)
    for i,tv_sets in enumerate(zip(tvs1,tvs2)):
        tvt['train'][i] = {}
        tvt['val'][i] = {}
        tv_set1 = tv_sets[0]
        tv_set2 = tv_sets[1]
        train_sets1, val_sets1 = SplitData(tv_set1, rs)
        train_sets2, val_sets2 = SplitData(tv_set2, rs)
        for j,train_set in enumerate(zip(train_sets1,train_sets2)):
            tvt['train'][i][j] = np.concatenate(train_set)
        for j,val_set in enumerate(zip(val_sets1,val_sets2)):
            tvt['val'][i][j] = np.concatenate(val_set)
    return(tvt)

#### selective kernel svm classifier within one cross validation
def SVM_CrossTrainAndPlot(trainsets, testsets, kernel, c_range, save_dst, pic_name, pic_title):###trainsets, testsets from CreateFoldsDataset
    print('===============%s'%kernel.upper(),'is being trained===============')
    ### split dataset into n folds
    n_folds = len(trainsets)
    n_c = len(c_range)
    rec_shape = (n_c,n_folds)
    tp_rec, tn_rec, fp_rec, fn_rec = [np.zeros(rec_shape) for i in range(4)]
    precision_means, recall_means, accuracy_means, f_value_means = [[] for i in range(4)]
    precision_stds, recall_stds, accuracy_stds, f_value_stds = [[] for i in range(4)]
    time_before = datetime.now()
    print('---', time_before, ': into C loop')
    accuracy_max= 0
    selected_c = c_range[0]
    c_id_now = 0
    f = codecs.open(save_dst+'information_rec.txt','w')
    f.write(kernel)
    f.write('\n')
    for c_index,Cvalue in enumerate(c_range):
        f.write(str(round(c_range[c_index], 1))+'\n')
        clf = svm.SVC(kernel=kernel, C=Cvalue)
        precision_rec,recall_rec,accuracy_rec,f_value_rec = [[] for i in range(4)]
        start_train = datetime.now()
        print('\n------------', start_train, ':', Cvalue, 'start training')
        for fold_index in range(n_folds):
            ### create train/test(validation)
            trainset = trainsets[fold_index]
            testset = testsets[fold_index]
            Xtrain = trainset[:, :-1]
            ytrain = trainset[:, -1]
            Xtest = testset[:, :-1]
            ytest = testset[:, -1]
            ### train
            clf.fit(Xtrain,ytrain)
            ### inference
            yHat = clf.predict(Xtest)
            ###record tp, tn, fp, fn
            tp, tn, fp, fn = EvaluateRes(ytest,yHat)
            tp_rec[c_index,fold_index] = tp
            tn_rec[c_index,fold_index] = tn
            fp_rec[c_index,fold_index] = fp
            fn_rec[c_index,fold_index] = fn
            ###record precision, recall, accuracy, f_value
            if tp != 0:
                precision = tp*1.0/(tp+fp)
            else:
                precision = 0
            recall = tp*1.0/(tp+fn)
            accuracy = (tp+tn)*1.0/(tp+tn+fp+fn)
            if recall!= 0 and precision != 0:
                f_value = 2*recall*precision/(recall+precision)
            else:
                f_value = 0
            precision_rec.append(precision);recall_rec.append(recall)
            accuracy_rec.append(accuracy);f_value_rec.append(f_value)
        end_train = datetime.now()
        print('------------', end_train, ': trained')
        c_train_time = end_train-start_train
        f.write('cost time: ' + str(c_train_time.total_seconds())+'\n')
        ### record n-fold mean
        precision_mean = np.mean(precision_rec);precision_std = np.std(precision_rec)
        precision_means.append(precision_mean);precision_stds.append(precision_std)
        recall_mean = np.mean(recall_rec);recall_std = np.std(recall_rec)
        recall_means.append(recall_mean);recall_stds.append(recall_std)
        f_value_mean = np.mean(f_value_rec);f_value_std = np.std(f_value_rec)
        f_value_means.append(f_value_mean);f_value_stds.append(f_value_std)
        accuracy_now = np.mean(accuracy_rec)
        if accuracy_now > accuracy_max:
            selected_c = Cvalue
            c_id_now = c_index
            accuracy_max = accuracy_now
        accuracy_mean = accuracy_now;accuracy_std = np.std(accuracy_rec)
        accuracy_means.append(accuracy_mean);accuracy_stds.append(accuracy_std)
        ### write evaluation into txt
        f.write(' '.join([str(round(precision_mean, 2)), str(round(recall_mean, 2)), str(round(f_value_mean, 2)), str(round(accuracy_mean, 2)), '\n\n']))
    ### save parametres
    np.savetxt(save_dst + 'tp.txt', tp_rec, fmt='%d')
    np.savetxt(save_dst + 'tn.txt', tn_rec, fmt='%d')
    np.savetxt(save_dst + 'fp.txt', fp_rec, fmt='%d')
    np.savetxt(save_dst + 'fn.txt', fn_rec, fmt='%d')
    time_after = datetime.now()
    print('\n---------',time_after, ': C loop completed')
    loop_cost = time_after - time_before
    print('---------','COST TIME: ',loop_cost.total_seconds(),'seconds')
    f.write('total cost: '+str(loop_cost.total_seconds()))
    f.close()
    f = codecs.open(save_dst + 'nfold_accuracy_rec.txt', 'w')
    for score in accuracy_means:
        f.write(str(score)+'\n')
    f.close()
    plt.plot(c_range, accuracy_means)
    plt.plot(c_range, precision_means)
    plt.plot(c_range, recall_means)
    plt.plot(c_range, f_value_means)
    plt.legend('accuracy_means precision_means recall_means f_value_means'.split())
    plt.title(pic_title+' kernel=%s'%kernel)
    plt.axvline(x=c_range[c_id_now], ymin=0, ymax=1, linewidth=1, color='k')
    plt.text(c_range[c_id_now], accuracy_max, '(' + str(round(c_range[c_id_now], 1)) + ', ' + str(accuracy_max) + ')')
    plt.savefig(save_dst + pic_name + '.png')
    plt.close()
    print('---------','plot completed')
    return (selected_c,accuracy_max)


### train model with selected_c using all training set
def SVM_TrainAndSave(TrainSet, kernel,C,save_dst):
    clf = svm.SVC(kernel=kernel, C=C)
    Xtrain = TrainSet[:, :-1]
    ytrain = TrainSet[:, -1]
    time_before_train = datetime.now()
    print('---------', time_before_train, ': max C completed')
    clf.fit(Xtrain, ytrain)
    time_after_train = datetime.now()
    train_cost = time_after_train - time_before_train
    print('---------', 'cost', train_cost.total_seconds(), 'for training')
    filename = save_dst + '.sav'
    joblib.dump(clf, filename)
    if kernel == 'linear':
        weight = clf.coef_[0]
        return (weight)
    else:
        return(0)

def SaveLinePlot(List,dst,pic_name,pic_title):
    plt.plot(List, '-o')
    plt.title(pic_title)
    for i, w in enumerate(List):
        plt.text(i, w, round(w, 3))
    plt.savefig(dst + pic_name + '.png')
    plt.close()
    print('weight plot completed')

def SaveAllLinePlot(Lists,dst,pic_name,pic_title):
    for list in Lists:
        plt.plot(list, '-o')
    plt.title(pic_title)
    plt.savefig(dst + pic_name + '.png')
    plt.close()
    print('weights plot in one completed')

def SVM_TrainAndPredict(trainSet, testset, kernel, C, save_dst):
    Xtrain = trainSet[:, :-1]
    ytrain = trainSet[:, -1]
    test_x = testset[:, :-1]
    test_y = testset[:, -1]
    clf = svm.SVC(kernel=kernel, C=C)
    time_before_train = datetime.now()
    print('---------', time_before_train, ': max C completed')
    clf.fit(Xtrain, ytrain)
    time_after_train = datetime.now()
    train_cost = time_after_train - time_before_train
    print('---------', 'cost', train_cost.total_seconds(), 'for training')
    filename = save_dst + '.sav'
    joblib.dump(clf, filename)
    y_hat = clf.predict(test_x)
    np.savetxt(save_dst + 'y_hat.txt', y_hat)
    test_accuracy = sum(y_hat == test_y) / len(test_y)
    return(test_accuracy)

def IntoTogether(lyric_data, lyric_size, non_lyric_data, non_lyric_size, dst, figure_title, c_range):
    f = codecs.open(dst + 'c_value.txt', 'w')
    for value in c_range:
        f.write(str(round(value, 1)) + '\n')
    f.close()
    random.seed(1)
    sample_times = 10
    ii = 0
    test_score_rec = {}
    for kernel in kernels:
        test_score_rec[kernel] = []
    while (ii < sample_times):
        lyric_data_id = random.sample(range(len(lyric_data)), lyric_size)
        use_lyric_dataset = lyric_data[lyric_data_id]
        rs_val = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        rs_test = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        notset_folds, test_folds, train_folds, val_folds = CreateValTrainsTests(use_lyric_dataset, non_lyric_data, rs_val,
                                                                                rs_test)
        result_rec = []
        for kernel in kernels:
            kernel_dst = dst + kernel + '/' + str(ii + 1)
            print('=========================================')
            figure_name = figure_title + '_'
            c, accuracy = SVM_CrossTrainAndPlot(train_folds, val_folds, kernel, c_range, kernel_dst, figure_name,
                                                figure_title)
            result_rec.append([kernel, c, accuracy])
            test_accuracy = SVM_TrainAndPredict(notset_folds[0], test_folds[0], kernel, c,
                                                kernel_dst + str(ii + 1) + '_')
            test_score_rec[kernel].append(test_accuracy)
        f = codecs.open(dst + 'result_rec_%s.txt' % (str(ii + 1)), 'w')
        for rec in result_rec:
            f.write(' '.join([str(a) for a in rec]) + '\n')
        f.close()
        ii += 1
        print(datetime.now())
        print(ii, 'over')
    for kernel in kernels:
        f = codecs.open(dst + kernel + '/test_score_rec.txt', 'w')
        for score in test_score_rec[kernel]:
            f.write(str(score) + '\n')
        f.close()
    print(datetime.now())
    print(figure_title, 'completed')


c_range = [round(0.1 * a,1) for a in range(1, 10)] + [1] + list(range(10, 151, 10))
kernels = ['linear','rbf', 'sigmoid']###poly太花时间了，暂且去掉
############################################################################################
########################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#################################
########################  take acc-line-trans as feature   #################################
########################^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#################################
############################################################################################
wiki_acc_src = '/home/liruihan/Desktop/data/wiki_entropy/wiki_accumulated_line_trans/'
lyric_acc_src = '/home/liruihan/Desktop/data/uni_accumulated_entropy_trans/'
wa_acc_src = '/home/liruihan/Desktop/data/wa_entropy/wa_accumulated_line_trans/'
cm_acc_src = '/home/liruihan/Desktop/data/cm_entropy/cm_accumulated_line_trans/'
aozora_acc_src = '/home/liruihan/Desktop/data/aozora_entropy/aozora_accumulated_line_trans/'

wiki_acc_line_trans = [e.split('\n')[:-1] for e in pre.read_data(wiki_acc_src)]
lyric_acc_line_trans = [e.split('\n')[:-1] for e in pre.read_data(lyric_acc_src,'num_')]
wa_acc_line_trans = [e.split('\n')[:-1] for e in pre.read_data(wa_acc_src)]
cm_acc_line_trans = [e.split('\n')[:-1] for e in pre.read_data(cm_acc_src)]
aozora_acc_line_trans = [e.split('\n')[:-1] for e in pre.read_data(aozora_acc_src)]


wiki_percentile_entropy = []
for trans in wiki_acc_line_trans:
    wiki_percentile_entropy.append(percentile_entropy(trans))

wa_percentile_entropy = []
for trans in wa_acc_line_trans:
    wa_percentile_entropy.append(percentile_entropy(trans))

lyric_percentile_entropy = []
for trans in lyric_acc_line_trans:
    lyric_percentile_entropy.append(percentile_entropy(trans))

cm_percentile_entropy = []
for trans in cm_acc_line_trans:
    cm_percentile_entropy.append(percentile_entropy(trans))

aozora_percentile_entropy = []
for trans in aozora_acc_line_trans:
    aozora_percentile_entropy.append(percentile_entropy(trans))
#########                      ######### #########         #########
### search when will the entropy down and mark with 1, else mark 0, accumulated percential
#########     #########   #########    #########
lyric_dataset = []
for data in lyric_percentile_entropy:
    OneOrZero = dict_transfer(data)
    GatheredDict = minus_in_percentile(OneOrZero)
    ProbPerPercentile = DictTransfer2Prob(GatheredDict)
    Values = list(ProbPerPercentile.values())
    lyric_dataset.append(Values+[1])

lyric_dataset = np.array(lyric_dataset)
####+++++++++++++++++++++++++++++++++++++++#####
wiki_dataset = []
for data in wiki_percentile_entropy:
    if data != {}:
        OneOrZero = dict_transfer(data)
        GatheredDict = minus_in_percentile(OneOrZero)
        ProbPerPercentile = DictTransfer2Prob(GatheredDict)
        Values = list(ProbPerPercentile.values())
        wiki_dataset.append(Values + [0])

wiki_dataset = np.array(wiki_dataset)
####+++++++++++++++++++++++++++++++++++++++#####
wa_dataset = []
for data in wa_percentile_entropy:
    if data != {}:
        OneOrZero = dict_transfer(data)
        GatheredDict = minus_in_percentile(OneOrZero)
        ProbPerPercentile = DictTransfer2Prob(GatheredDict)
        Values = list(ProbPerPercentile.values())
        wa_dataset.append(Values + [0])

wa_dataset = np.array(wa_dataset)
####+++++++++++++++++++++++++++++++++++++++#####
cm_dataset = []
for data in cm_percentile_entropy:
    if data != {}:
        OneOrZero = dict_transfer(data)
        GatheredDict = minus_in_percentile(OneOrZero)
        ProbPerPercentile = DictTransfer2Prob(GatheredDict)
        Values = list(ProbPerPercentile.values())
        cm_dataset.append(Values + [0])

cm_dataset = np.array(cm_dataset)
####+++++++++++++++++++++++++++++++++++++++#####
aozora_dataset = []
for data in aozora_percentile_entropy:
    OneOrZero = dict_transfer(data)
    GatheredDict = minus_in_percentile(OneOrZero)
    ProbPerPercentile = DictTransfer2Prob(GatheredDict)
    Values = list(ProbPerPercentile.values())
    aozora_dataset.append(Values+[0])

aozora_dataset = np.array(aozora_dataset)
################ lyric_watanabe ####################                          @@@@@@@@@@@@^^^^^^^^^^^^^^^
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
print('------------------acc01 START: lyric_watanabe')

########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
dst = '/home/liruihan/Desktop/result/SVM_Model_Selection/lyric_watanabe/acc_01/'#same sample size
f = codecs.open(dst+'c_value.txt','w')
for value in c_range:
    f.write(str(round(value,1))+'\n')
f.close()
figure_title = 'watanabe'

lyric_size = len(wa_dataset)
wa_size = len(wa_dataset)
f = codecs.open(dst+'data_infomation.txt','w')
f.write('lyric size: '+str(lyric_size)+'\n'
        + 'wa size: ' + str(wa_size) + '\n')
f.close()
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
random.seed(0)
lyric_data_id = random.sample(range(len(lyric_dataset)), lyric_size)
use_lyric_dataset = lyric_dataset[lyric_data_id]
rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
tvt = CreateFoldsDataset(use_lyric_dataset, wa_dataset, rs)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
result_rec = []
### training set for all
for kernel in kernels:
    test_score_rec = []
    for i in range(len(tvt['test'])):
        kernel_dst = dst + kernel + '/' + str(i+1) + '_'
        testSet = tvt['test'][i]
        trainSet = np.concatenate((tvt['train'][i][0], tvt['val'][i][0]), axis=0)
        print('=========================================')
        figure_name = figure_title+'_'+str(i + 1)
        ### SVM_CrossTrainAndPlot(trainsets, testsets, kernel, c_range, save_dst, pic_name, pic_title):###trainsets, testsets from CreateFoldsDataset
        c, accuracy = SVM_CrossTrainAndPlot(tvt['train'][i], tvt['val'][i], kernel, c_range, kernel_dst, figure_name, figure_title)
        result_rec.append([kernel, c, accuracy])
        ### SVM_TrainAndSave(TrainSet, kernel,C,save_dst)
        test_score = SVM_TrainAndPredict(trainSet, testSet, kernel,c,kernel_dst)
        test_score_rec.append(test_score)
    f = codecs.open(dst+kernel+'/test_score_rec.txt','w')
    for score in test_score_rec:
        f.write(str(score)+'\n')

f = codecs.open(dst+'result_rec.txt','w')
for rec in result_rec:
    f.write(' '.join([str(a) for a in rec])+'\n')

f.close()
print(datetime.now())
print(figure_title,'completed')
################ lyric_cm ####################
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
print('------------------acc01 START: lyric_cm')

########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
dst = '/home/liruihan/Desktop/result/SVM_Model_Selection/lyric_cm/acc_01/'
f = codecs.open(dst+'c_value.txt','w')
for value in c_range:
    f.write(str(round(value,1))+'\n')
f.close()
figure_title = 'cm'

lyric_size = len(cm_dataset)
cm_size = len(cm_dataset)
f = codecs.open(dst+'data_infomation.txt','w')
f.write('lyric size: '+str(lyric_size)+'\n'
        + 'cm size: ' + str(cm_size) + '\n')
f.close()
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
random.seed(0)
lyric_data_id = random.sample(range(len(lyric_dataset)), lyric_size)
use_lyric_dataset = lyric_dataset[lyric_data_id]
rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
tvt = CreateFoldsDataset(use_lyric_dataset, cm_dataset, rs)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
result_rec = []
### training set for all
for kernel in kernels:
    test_score_rec = []
    for i in range(len(tvt['test'])):
        print(kernel)
        print('{}th fold in k-fold'.format(i))
        kernel_dst = dst + kernel + '/' + str(i+1) + '_'
        testSet = tvt['test'][i]
        trainSet = np.concatenate((tvt['train'][i][0], tvt['val'][i][0]), axis=0)
        print('=========================================')
        figure_name = figure_title+'_'+str(i + 1)
        ### SVM_CrossTrainAndPlot(trainsets, testsets, kernel, c_range, save_dst, pic_name, pic_title):###trainsets, testsets from CreateFoldsDataset
        c, accuracy = SVM_CrossTrainAndPlot(tvt['train'][i], tvt['val'][i], kernel, c_range, kernel_dst, figure_name, figure_title)
        result_rec.append([kernel, c, accuracy])
        ### SVM_TrainAndSave(TrainSet, kernel,C,save_dst)
        test_score = SVM_TrainAndPredict(trainSet, testSet, kernel,c,kernel_dst)
        test_score_rec.append(test_score)
    f = codecs.open(dst+kernel+'/test_score_rec.txt','w')
    for score in test_score_rec:
        f.write(str(score)+'\n')

f = codecs.open(dst+'result_rec.txt','w')
for rec in result_rec:
    f.write(' '.join([str(a) for a in rec])+'\n')

f.close()
print(datetime.now())
print(figure_title,'completed')

################ lyric_wikipedia ####################
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
print('------------------acc01 START: lyric_wikipedia')

########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
dst = '/home/liruihan/Desktop/result/SVM_Model_Selection/lyric_wikipedia/acc_01/'
f = codecs.open(dst+'c_value.txt','w')
for value in c_range:
    f.write(str(round(value,1))+'\n')
f.close()
figure_title = 'wiki'

lyric_size = len(wiki_dataset)
wiki_size = len(wiki_dataset)
f = codecs.open(dst+'data_infomation.txt','w')
f.write('lyric size: '+str(lyric_size)+'\n'
        + 'wiki size: ' + str(wiki_size) + '\n')
f.close()
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
random.seed(0)
lyric_data_id = random.sample(range(len(lyric_dataset)), lyric_size)
use_lyric_dataset = lyric_dataset[lyric_data_id]
rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
tvt = CreateFoldsDataset(use_lyric_dataset, wiki_dataset, rs)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
result_rec = []
### training set for all
for kernel in kernels:
    test_score_rec = []
    for i in range(len(tvt['test'])):
        print(kernel)
        print('{}th fold in k-fold'.format(i))
        kernel_dst = dst + kernel + '/' + str(i+1) + '_'
        testSet = tvt['test'][i]
        trainSet = np.concatenate((tvt['train'][i][0], tvt['val'][i][0]), axis=0)
        print('=========================================')
        figure_name = figure_title+'_'+str(i + 1)
        ### SVM_CrossTrainAndPlot(trainsets, testsets, kernel, c_range, save_dst, pic_name, pic_title):###trainsets, testsets from CreateFoldsDataset
        c, accuracy = SVM_CrossTrainAndPlot(tvt['train'][i], tvt['val'][i], kernel, c_range, kernel_dst, figure_name, figure_title)
        result_rec.append([kernel, c, accuracy])
        ### SVM_TrainAndSave(TrainSet, kernel,C,save_dst)
        test_score = SVM_TrainAndPredict(trainSet, testSet, kernel,c,kernel_dst)
        test_score_rec.append(test_score)
    f = codecs.open(dst+kernel+'/test_score_rec.txt','w')
    for score in test_score_rec:
        f.write(str(score)+'\n')

f = codecs.open(dst+'result_rec.txt','w')
for rec in result_rec:
    f.write(' '.join([str(a) for a in rec])+'\n')

f.close()
print(datetime.now())
print(figure_title,'completed')
################ lyric_wikipedia ####################
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
print('------------------acc01 START: lyric_wikipedia')

########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
dst = '/home/liruihan/Desktop/result/SVM_Model_Selection/lyric_aozora/acc_01/'
f = codecs.open(dst+'c_value.txt','w')
for value in c_range:
    f.write(str(round(value,1))+'\n')
f.close()
figure_title = 'aozora'

lyric_size = len(aozora_dataset)
aozora_size = len(aozora_dataset)
f = codecs.open(dst+'data_infomation.txt','w')
f.write('lyric size: '+str(lyric_size)+'\n'
        + 'wiki size: ' + str(aozora_size) + '\n')
f.close()
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
random.seed(0)
lyric_data_id = random.sample(range(len(lyric_dataset)), lyric_size)
use_lyric_dataset = lyric_dataset[lyric_data_id]
rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
tvt = CreateFoldsDataset(use_lyric_dataset, aozora_dataset, rs)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
result_rec = []
### training set for all
for kernel in kernels:
    test_score_rec = []
    for i in range(len(tvt['test'])):
        print(kernel)
        print('{}th fold in k-fold'.format(i))
        kernel_dst = dst + kernel + '/' + str(i+1) + '_'
        testSet = tvt['test'][i]
        trainSet = np.concatenate((tvt['train'][i][0], tvt['val'][i][0]), axis=0)
        print('=========================================')
        figure_name = figure_title+'_'+str(i + 1)
        ### SVM_CrossTrainAndPlot(trainsets, testsets, kernel, c_range, save_dst, pic_name, pic_title):###trainsets, testsets from CreateFoldsDataset
        c, accuracy = SVM_CrossTrainAndPlot(tvt['train'][i], tvt['val'][i], kernel, c_range, kernel_dst, figure_name, figure_title)
        result_rec.append([kernel, c, accuracy])
        ### SVM_TrainAndSave(TrainSet, kernel,C,save_dst)
        test_score = SVM_TrainAndPredict(trainSet, testSet, kernel,c,kernel_dst)
        test_score_rec.append(test_score)
    f = codecs.open(dst+kernel+'/test_score_rec.txt','w')
    for score in test_score_rec:
        f.write(str(score)+'\n')

f = codecs.open(dst+'result_rec.txt','w')
for rec in result_rec:
    f.write(' '.join([str(a) for a in rec])+'\n')

f.close()
print(datetime.now())
print(figure_title,'completed')
################ lyric_lyric ####################
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
print('------------------acc01 START: lyric_lyric')

########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
dst = '/home/liruihan/Desktop/result/SVM_Model_Selection/lyric_lyric/acc_01/'
f = codecs.open(dst+'c_value.txt','w')
for value in c_range:
    f.write(str(round(value,1))+'\n')
f.close()
figure_title = 'lyric'

lyric_all_size = len(lyric_dataset)
lable1_ratio = 0.5
lyric1_size = int(lyric_all_size*lable1_ratio)
lyric0_size = lyric_all_size-lyric1_size
f = codecs.open(dst+'data_infomation.txt','w')
f.write('lyric labeled 1 size: '+str(lyric1_size)+'\n'
        + 'lyric labeled 0 size: ' + str(lyric0_size) + '\n')
f.close()

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
random.seed(0)
for_label = random.sample(lyric_dataset, lyric_all_size)
lyric_dataset_1 = for_label[:lyric1_size]
lyric_dataset_0 = [a[:-1] + [0] for a in for_label if a not in for_label[lyric1_size:]]
rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
########### CHANGE #######################
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
tvt = CreateFoldsDataset(lyric_dataset_1, lyric_dataset_0, rs)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########### CHANGE #######################
result_rec = []
### training set for all
for kernel in kernels:
    test_score_rec = []
    for i in range(len(tvt['test'])):
        print(kernel)
        print('{}th fold in k-fold'.format(i))
        kernel_dst = dst + kernel + '/' + str(i+1) + '_'
        testSet = tvt['test'][i]
        trainSet = np.concatenate((tvt['train'][i][0], tvt['val'][i][0]), axis=0)
        print('=========================================')
        figure_name = figure_title+'_'+str(i + 1)
        ### SVM_CrossTrainAndPlot(trainsets, testsets, kernel, c_range, save_dst, pic_name, pic_title):###trainsets, testsets from CreateFoldsDataset
        c, accuracy = SVM_CrossTrainAndPlot(tvt['train'][i], tvt['val'][i], kernel, c_range, kernel_dst, figure_name, figure_title)
        result_rec.append([kernel, c, accuracy])
        ### SVM_TrainAndSave(TrainSet, kernel,C,save_dst)
        test_score = SVM_TrainAndPredict(trainSet, testSet, kernel,c,kernel_dst)
        test_score_rec.append(test_score)
    f = codecs.open(dst+kernel+'/test_score_rec.txt','w')
    for score in test_score_rec:
        f.write(str(score)+'\n')

f = codecs.open(dst+'result_rec.txt','w')
for rec in result_rec:
    f.write(' '.join([str(a) for a in rec])+'\n')

f.close()
print(datetime.now())
print(figure_title,'completed')
