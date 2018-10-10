from simhash import Simhash 
import numpy as np 
import math
import Levenshtein as ls
import pandas as pd 


class SimWord(object):
    """docstring for SimWord"""
    def __init__(self):
        pass
        
    def simhash_func(self,str1,str2):
        '''
        simhash计算唯一性
        '''
        res =  Simhash(str1).distance(Simhash(str2))
        return res

    def numsim_func(self,num1,num2):
        '''
        数值大小唯一性识别，比如0.87与0.88是非常接近的
        '''
        mid = (num1 + num2)/2.0
        numres = np.sqrt((math.pow(num1-mid,2)+math.pow(num2-mid,2)) / 2.0) / mid
        return numres

    def jaro_winkler_func(self,str1,str2):
        '''
        编辑距离jaro_winkler算法
        参考https://www.cnblogs.com/zangrunqiang/p/6752430.html
        针对多字错字错位的情况相似度特高测试代码dis_test.py效果挺好
        '''
        res = ls.jaro_winkler(str1,str2)
        return res

    def dislocation_func(self,str1,str2):
        '''
        编辑距离jaro_winkler算法
        参考https://www.cnblogs.com/zangrunqiang/p/6752430.html
        针对多字错字错位的情况相似度效果不佳
        '''   
        res = ls.ratio(str1,str2)
        return res

    def distance_func(self,str1,str2):
        '''
        编辑距离jaro_winkler算法
        参考https://www.cnblogs.com/zangrunqiang/p/6752430.html
        针对多字错字错位的情况相似度效果不佳
        '''
        res = ls.distance(str1,str2)
        return res

def read_datas():
    with open('hyper_doc2vec/datasets/bot_question.txt','r',encoding='utf-8') as f:
        data = f.readlines() 
    return data

def jaro_winkler_test(str2):
    data = read_datas()
    max_dis = 0.9
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.jaro_winkler_func(str1,str2)
        #print(jwres)
        #res = str1 if jwres < 0.05 else str2
        #print (res)
        #return res
        if jwres > max_dis:
            max_dis = jwres
            print (str1)
            return str1

def simhash_test(str2):
    data = read_datas()
    min_dis = 0.1
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.simhash_func(str1,str2)
        #print(jwres)
        #res = str1 if jwres < 0.05 else str2
        #print (res)
        #return res
        if jwres < min_dis:
            min_dis = jwres
            print (str1)
            return str1

def dislocation_test(str2):
    data = read_datas()
    min_dis = 0.05
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.dislocation_func(str1,str2)
        #print(jwres)
        #res = str1 if jwres < 0.05 else str2
        #print (res)
        #return res
        if jwres < min_dis:
            min_dis = jwres
            print (str1)
            return str1

def distance_test(str2):
    data = read_datas()
    min_dis = 0.05
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.distance_func(str1,str2)
        #print(jwres)
        #res = str1 if jwres < 0.05 else str2
        #print (res)
        #return res
        if jwres < min_dis:
            min_dis = jwres
            print (str1)
            return str1

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

def jaro_winkler_sim(str2):
    data = read_datas()
    max_dis = 0.9
    list1 = []
    list2 = []
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.jaro_winkler_func(str1,str2)

        #if jwres > max_dis:
        #    list1.append(jwres)
        #    list2.append(str1)
        list1.append(jwres)
        list2.append(str1)
    max_x(list1,list2)
 
def simhash_sim(str2):
    data = read_datas()
    min_dis = 0.1
    list1 = []
    list2 = []
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.simhash_func(str1,str2)
        #if jwres > max_dis:
        #    list1.append(jwres)
        #    list2.append(str1)
        list1.append(jwres)
        list2.append(str1)
    min_x(list1,list2)

def dislocation_sim(str2):
    data = read_datas()
    min_dis = 0.05
    list1 = []
    list2 = []
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.dislocation_func(str1,str2)
        #if jwres > max_dis:
        #    list1.append(jwres)
        #    list2.append(str1)
        list1.append(jwres)
        list2.append(str1)
    min_x(list1,list2)

def distance_sim(str2):
    data = read_datas()
    min_dis = 0.05
    list1 = []
    list2 = []
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.distance_func(str1,str2)
        #if jwres > max_dis:
        #    list1.append(jwres)
        #    list2.append(str1)
        list1.append(jwres)
        list2.append(str1)
    min_x(list1,list2)

def simhash_x(str2):
    data = read_datas()
    min_dis = 0.1
    sim_list = []
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.simhash_func(str1,str2)
        sim_list.append(jwres)
    indx = np.argmin(sim_list)
    for j in range(47592):
        res = data[j].strip()
        if j == indx:
            print (f'{res}:{sim_list[indx]}')
            return res

def main():
    str1 = '高血压，动脉硬化失眠.，怎么办？'
    str2 = '高血压，动脉硬化失眠，怎么办？'
    str3 = '朋友是高血压引发的肾病患者，想知道高血压导致的肾病的患者需要怎么饮食？'
    str4 = '朋友是高血压引发的肾病患者，想知道高血压引发的肾病患者的危害有哪些呢？'
    str5 = '失眠会引起高血压吗'
    str6 = '高血压者偶尔失眠怎么办？'
    str7 = '高血压心脏病能做飞机吗'
    str8 = '高血压心脏病能坐飞机吗'
    simhash_x(str1)
    simhash_x(str2)
    simhash_x(str5)
    '''
    print('str1 and str2')
    jaro_winkler_test(str1,str2)
    simhash_test(str1,str2)
    dislocation_test(str1,str2)
    distance_test(str1,str2)
    print('str3 and str4')
    jaro_winkler_test(str3,str4)
    simhash_test(str3,str4)
    dislocation_test(str3,str4)
    distance_test(str3,str4)
    print('str5 and str6')
    jaro_winkler_test(str5,str6)
    simhash_test(str5,str6)
    dislocation_test(str5,str6)
    distance_test(str5,str6)
    print('str7 and str8')
    jaro_winkler_test(str7,str8)
    simhash_test(str7,str8)
    dislocation_test(str7,str8)
    distance_test(str7,str8)
    
    print('str1')
    jaro_winkler_test(str1)
    simhash_test(str1)
    dislocation_test(str1)
    distance_test(str1)
    print('str2')
    jaro_winkler_test(str2)
    simhash_test(str2)
    dislocation_test(str2)
    distance_test(str2)
    print('str3')
    jaro_winkler_test(str3)
    simhash_test(str3)
    dislocation_test(str3)
    distance_test(str3)
    print('str4')
    jaro_winkler_test(str4)
    simhash_test(str4)
    dislocation_test(str4)
    distance_test(str4)
    print('str5 and str6')
    jaro_winkler_test(str6)
    simhash_test(str6)
    dislocation_test(str6)
    distance_test(str6)
    print('str7 and str8')
    jaro_winkler_test(str8)
    simhash_test(str8)
    dislocation_test(str8)
    distance_test(str8)
    print('*******************************************')
    print('str1')
    jaro_winkler_sim(str1)
    simhash_sim(str1)
    dislocation_sim(str1)
    distance_sim(str1)
    print('str2')
    jaro_winkler_sim(str2)
    simhash_sim(str2)
    dislocation_sim(str2)
    distance_sim(str2)
    print('str3')
    jaro_winkler_sim(str3)
    simhash_sim(str3)
    dislocation_sim(str3)
    distance_sim(str3)
    print('str4')
    jaro_winkler_sim(str4)
    simhash_sim(str4)
    dislocation_sim(str4)
    distance_sim(str4)
    print('str5 and str6')
    jaro_winkler_sim(str6)
    simhash_sim(str6)
    dislocation_sim(str6)
    distance_sim(str6)
    print('str7 and str8')
    jaro_winkler_sim(str8)
    simhash_sim(str8)
    dislocation_sim(str8)
    distance_sim(str8)
    '''
    print('Finished!!!')
    
if __name__ == '__main__':
    main()
