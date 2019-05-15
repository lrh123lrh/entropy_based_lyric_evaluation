#######################################################################
#######################################################################
##############################Explanation##############################
##############################.._str: data in string type
##############################.._int: data in int type
##############################.._list: data in list type
#######################################################################
#######################################################################

import codecs
import MeCab as mecab
import re

###import raw lyric###
def read_data(dirpath, label = '', start = 1):
    dataset = []
    while True:
        try:
            path = dirpath + label + '%d.txt'%start
            data = codecs.open(path).read()
            dataset.append(data)
            start += 1
        except IOError:
            NumLyrics = start - 1
            print(NumLyrics)
            break
    return dataset

###split information###
def lyric_info1(lyric):###useless anymore(20180831)
    eles = lyric.split('\n\n', 2)
    title = eles[0]
    man = eles[1].split('\n')
    singer = man[0]
    songwriter = man[1][3:]
    composer = man[2][3:]
    lyric = eles[2]
    return title, singer, songwriter, composer, lyric

#####################______________________####################
###############################################################
############extract title, singer, songwriter, composer, lyric informaton
############from one lyric data in UniqueDataset
############Input: lyric_data_str
############Output: [title_str, singer_str, songwriter_str, composer_str, lyric_str]
###############################################################
def lyric_info(lyric):
    eles = lyric.split('\n\n', 4)
    title = eles[0]
    singer = eles[1]
    songwriter = eles[2]
    composer = eles[3]
    lyric = eles[4]
    return title, singer, songwriter, composer, lyric
###############################################################
#####################______________________####################

#####################______________________####################
###############################################################
############extract title, singer, songwriter, composer, lyric informaton
############from all lyric data in UniqueDataset
############Input: lyric_data_list = [lyric_data_0_str,lyric_data_1_str, ...]
############Output: [title_list, singer_list, songwriter_list, composer_list, lyric_list]
############e.x.: lyric_list = [lyric_0_str, lyric_1_str, ...]
###############################################################
def dataset_info(dataset):
    eles = [lyric.split('\n\n',4) for lyric in dataset]
    title = [ele[0] for ele in eles]
    singer = [ele[1] for ele in eles]
    songwriter = [ele[2] for ele in eles]
    composer = [ele[3] for ele in eles]
    lyric = [ele[4] for ele in eles]
    return title, singer, songwriter, composer, lyric
###############################################################
#####################______________________####################


#####################______________________####################
###############################################################
############complete lyric in UniqueDataset
############Input: lyric_str
############Output: lyric_str
###############################################################
def complete(text):###return str
    lyric_block=text.split('\n\n')
    #print(lyric_block)
    target=['(※くり返し)','(△くり返し)','(□くり返し)','(※くりかえし)','(△くりかえし)','(□くりかえし)']
    #labelset=['※','△','□']
    pair=list(enumerate(lyric_block))
    for t in target:
        #print(t)
        if(t in text):
            #label=labelset[target.index(t)%3]
            label=t[1]
            target_id=[p[0] for p in pair if t in p[1]]#繰り返し矢印の
            #print(target_id)
            reid=[p[0] for p in pair if label in p[1] and p[0] not in target_id]#繰り返し部分の
            #print(reid)
            for a in reid:
                #print(lyric_block[a])
                lyric_block[a]=lyric_block[a].replace(label,'')
            if (len(reid)==2):
                re_part_list=lyric_block[reid[0]:reid[1]+1]
                re_part_str='\n\n'.join(re_part_list)
            elif (len(reid)==1):
                re_part_str=lyric_block[reid[0]]
            else:
                re_part_str=''
            for i in target_id:
                if(lyric_block[i] == t+'\n'):
                    lyric_block[i]=re_part_str
                elif (lyric_block[i] != t+'\n'):
                    lyric_block[i]=lyric_block[i].replace(t,re_part_str+'\n')
        else:
            continue
    completed_text='\n\n'.join(lyric_block)
    return completed_text
###############################################################
#####################______________________####################


#####################______________________####################
###############################################################
############ preprocess data in corpus folder
class parse_corpus_lyric:
    def __init__(self,corpus_lyric,ifpos = True,iforigin = True):
        self.lyric_lines = []
        self.lyric_words = []
        self.lyric_para_line = []
        paragraphs_info = corpus_lyric.split('\n\n\n')[:-1]
        for paragraph_info in paragraphs_info:
            lines = []
            lines_info = paragraph_info.split('\n\n')
            for line_info in lines_info:
                words = []
                words_info = line_info.split('\n')
                for word_info in words_info:
                    infos = word_info.split(' ')
                    pos = self.extract_pos(infos)
                    if(iforigin):
                        word = self.extract_origin_word(infos)
                    else:
                        word = self.extract_word(infos)
                    print(word)
                    if word == '' or pos == '':
                        continue
                    if(ifpos):
                        pair = (word, pos)
                        self.lyric_words.append(pair)
                        words.append(pair)
                    else:
                        self.lyric_words.append(word)
                        words.append(word)
                lines.append(words)
                self.lyric_lines.append(words)
            self.lyric_para_line.append(lines)
    def extract_origin_word(self,info):
        if (info != ['']):
            stem = info[7]
            origin = info[0]
            if stem != '*':
                return(stem)
            else:
                return(origin)
        else:
            return('')
    def extract_word(self,info):
        if (info != ['']):
            origin = info[0]
            return(origin)
        else:
            return('')
    def extract_pos(self,info):
        if (info != ['']):
            return(info[1])
        else:
            return('')
###############################################################
#####################______________________####################

#####################______________________####################
###############################################################
###choose word whose pos is in non_stop_pos
def drop_stop_word(word_pos_list, non_stop_pos = ['フィラー','その他','福祉','助詞','形容詞',
                '助動詞','動詞','名刺','感動詞','接続詞','接頭詞','連体詞','英語']):
    non_stop_words = []
    for pair in word_pos_list:
        if pair[1] in non_stop_pos:
            non_stop_words.append(pair[0])
    return(non_stop_words)

###############################################################
#####################______________________####################


def NofBlocks2id(n, SegmentedData):
    return(SegmentedData.index(ly) for ly in SegmentedData if len(ly) == n)

def NofSentencesInBlock2id(s_n, b_id, SegmentedData):
    while True:
        try:
            return([SegmentedData.index(ly) for ly in SegmentedData if len(ly[b_id-1].split('\n')) == s_n])
        except IndexError:
            print('do not have blocks more than %s'%str(b_id-1))
            break

def id2NofBlocks(id, SegmentedData):
    segmented = [SegmentedData[a] for a in id]
    return([len(a) for a in segmented])

def id2NofSentencesInBlock(id, b_id, SegmentedData):
    segmented = [SegmentedData[a] for a in id]
    return([len(a[b_id-1].split('\n')) for a in segmented])

def Average(List):
    return(sum(List)/len(List))

def ProbDist(data, EleId):
    count = {}
    for listt in data:
        if(listt[EleId] not in count.keys()):
            count[listt[EleId]] = 1
        else:
            count[listt[EleId]] +=1
    for K in count.keys():
        count[K] /= len(data)
    return(count)

def ListEle(List,EleId):###List is list, EleId is list
    ele =''
    if(len(EleId) > 1):
        for i in EleId:
            ele += str(List[i]) + ' '
        return(ele[:-1])
    else:
        return(str(List[EleId[0]]))

def CondiProbDist(data, CondiEleId, TargetEleId):###return{'parent11 parent12 ... parent1n':child1,...}
    count = {}
    for listt in data:
        K = ListEle(listt, CondiEleId)
        if(K not in count.keys()):
            count[K] = {}
            count[K][listt[TargetEleId]] = 1
        else:
            T = listt[TargetEleId]
            if(T not in count[K].keys()):
                count[K][T] = 1
            else:
                count[K][T] +=1
    for K in count.keys():
        Total = sum(count[K].values())
        for KK in count[K].keys():
            count[K][KK] /= Total
    return(count)

def WordProb(word, text):###word prob in text(word list)
    all = len(text)
    some = text.count(word)
    prob = some/all
    return(prob)##return a float number

def ParsedText(text):###parse text with mecab and hold line break/eol
    parser = mecab.Tagger('-Owakati')
    parsed = ''
    text2para = text.split('\n\n')
    for para in text2para:
        para2line = para.split('\n')
        for line in para2line:
            Wordlist = parser.parse(line)
            parsed += Wordlist
        parsed += '\n'
    return(parsed)

def YomiText(text):###parse text with mecab and hold line break/eol
    parser = mecab.Tagger('-Oyomi')
    parsed = ''
    text2para = text.split('\n\n')
    for para in text2para:
        para2line = para.split('\n')
        for line in para2line:
            Wordlist = parser.parse(line)
            parsed += Wordlist
        parsed += '\n'
    return(parsed)

def AllStemText(text):
    parser = mecab.Tagger()
    parsed = ''
    text2para = text.split('\n\n')
    for para in text2para:
        para2line = para.split('\n')
        for line in para2line:
            words_info = parser.parse(line).split('\n')
            temp = []
            for info in words_info[:-2]:
                info = re.split('[,\t]',info)
                stem = info[7]
                origin = info[0]
                if stem != '*':
                    temp.append(stem)
                else:
                    temp.append(origin)
            parsed += ' '.join(temp) + '\n'
        parsed += '\n'
    return(parsed)

def remove_values(the_list, val):
    while val in the_list:
        the_list.remove(val)
