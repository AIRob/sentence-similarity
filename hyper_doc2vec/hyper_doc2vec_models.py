# -*- coding: utf-8 -*-

import sys
import gensim
import sklearn
import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import codecs
import numpy
import jieba
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
TaggededDocument = gensim.models.doc2vec.TaggedDocument
jieba.load_userdict("./user_disease_hyper_dict/my_hyper_dict.txt")


#bot_qa.txt bot_question.txt
def get_datasest():
    with open("./datasets/bot_qa.txt", 'r',encoding='utf-8', errors='ignore') as cf:
        docs = cf.readlines()
        print (len(docs))
    x_train = []
    #y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train

def train_doc2vec(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train,min_count=1, window = 3, size = size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    #hyper_bot_qa.doc2vec hyper_bot_question.doc2vec
    model_dm.save('brain/hyper_bot_qa.doc2vec')
    return model_dm

def test_doc2vec():
    model_hdd = Doc2Vec.load("brain/hyper_bot_qa.doc2vec")
    test_text = ['高血压','高血脂', '血压']
    inferred_vector_hdd = model_hdd.infer_vector(test_text)
    print (inferred_vector_hdd)
    sims = model_hdd.docvecs.most_similar([inferred_vector_hdd], topn=10)
    return sims

start_alpha = 0.01
infer_epoch = 1000
docvec_size = 192

def calc_simlarity(vector1, vector2):
    vector1mod = np.sqrt(vector1.dot(vector1))
    vector2mod = np.sqrt(vector2.dot(vector2))
    if vector2mod != 0 and vector1mod != 0:
        simlarity = (vector1.dot(vector2)) / (vector1mod * vector2mod)
    else:
        simlarity = 0
    return simlarity

def doc2vec(filename, model):
    doc = [w for x in codecs.open(filename, 'r', 'utf-8').readlines() for w in jieba.cut(x.strip())]
    doc_vec_all = model.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)
    return doc_vec_all

def doc2vec_train():
    x_train = get_datasest()
    train_doc2vec(x_train)

def read_datas():
    with open('datasets/bot_question.txt','r',encoding='utf-8') as f:
        data = f.readlines() 
    return data

def main2(str2,model):
    data = read_datas()
    sim_list = []
    sim_listx = []
    max_dis = 0.9
    for i in range(47592):
        str1 = data[i].strip()
        
        #str2 = '失眠会？高血压吗引起'
        sentence2vec1 = sentence2vec(str1, model)
        sentence2vec2 = sentence2vec(str2, model)
        #print(calc_simlarity(sentence2vec1, sentence2vec2))
        sim_res = calc_simlarity(sentence2vec1, sentence2vec2)
        sim_list.append(sim_res)
        if sim_res > max_dis:
            max_dis = sim_res
            sim_listx.append(sim_res)
            print (f'{str1}:{sim_res}')
            #return str1
    print(max(sim_list))
    print(max(sim_listx))

def max_x(list1,list2):
    if len(list1) == len(list2):
        indx = np.argmax(list1)
        res = list2[indx]
        print(f'{res}:{list1[indx]}')
        return res 

def min_x(list1,list2):
    if len(list1) == len(list2):
        indx = np.argmin(list1)
        res = list2[indx]
        print(f'{res}:{list1[indx]}')
        return res   

def main3(str2,model):
    data = read_datas()
    sim_list = []
    sim_listx = []
    max_dis = 0.9
    list1 = []
    list2 = []
    for i in range(47592):
        str1 = data[i].strip()
        
        #str2 = '失眠会？高血压吗引起'
        sentence2vec1 = sentence2vec(str1, model)
        sentence2vec2 = sentence2vec(str2, model)
        #print(calc_simlarity(sentence2vec1, sentence2vec2))
        sim_res = calc_simlarity(sentence2vec1, sentence2vec2)
        #sim_list.append(sim_res)
        list1.append(sim_res)
        list2.append(str1)
    max_x(list1,list2)

def read_datas():
    with open('datasets/bot_question.txt','r',encoding='utf-8') as f:
        data = f.readlines() 
    return data

def sentence2vec(sentence, model):
    doc = [w for w in jieba.cut(sentence.strip())]
    doc_vec_all = model.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)
    return doc_vec_all

def main(n,model):

    #model = gensim.models.Word2Vec.load('./data/hyper_datadisease.word2vec')
    filename1 = f'datasets/{n}/{n}1.txt'
    filename2 = f'datasets/{n}/{n}2.txt'
    doc2vec1 = doc2vec(filename1, model)
    doc2vec2 = doc2vec(filename2, model)
    print(calc_simlarity(doc2vec1, doc2vec2))

if __name__ == '__main__':
    #doc2vec_train()
    str2 = '失眠会？高血压吗引起'
    str1 = '失眠会高血压吗引起？'
    model1 = Doc2Vec.load("brain/hyper_bot_question.doc2vec")
    model2 = Doc2Vec.load("brain/hyper_bot_qa.doc2vec")
    model3 = Doc2Vec.load('brain/hyper_hyperqa.doc2vec')
    model4 = Doc2Vec.load('brain/hyper_answer.doc2vec')
    model5 = Doc2Vec.load('brain/hyper_datadiseaseX.doc2vec')
    '''
    main('P',model1)
    main('a',model1)
    main('q',model1)
    main('s',model1)
    
    main('P',model2)
    main('a',model2)
    main('q',model2)
    main('s',model2)
    
    main('P',model3)
    main('a',model3)
    main('q',model3)
    main('s',model3)

    main('P',model4)
    main('a',model4)
    main('q',model4)
    main('s',model4)

    main('P',model5)
    main('a',model5)
    main('q',model5)
    main('s',model5)
    '''
    #main2(model1)
    #main2(model2)

    main2(str1,model1)
    main2(str1,model2)

    #main2(str2,model1)
    #main2(str2,model2)
    
    main3(str2,model1)
    main3(str2,model2)