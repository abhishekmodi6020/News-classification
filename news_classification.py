# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:33:20 2019

@author: Abhishek Modi
"""
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def load_text_data():
    # Stores the training data and the corresponding ouptut class
    train_data = []
#    # stores the output class of the corresponding training data
#    train_class = []
    # Stores the testing data and the corresponding ouptut class
    test_data = []
#    # stores the output class of the corresponding testing data
#    test_class = []
    # Stores Unique output_classes.
    output_class = {}
#    output_class_name = []
    
    # Reading from each file from each class folder and storing it in the above intitialization.
    count = 0
    for each_class in sorted(os.listdir("20_newsgroups")):
        output_class[each_class] = count
        count += 1
    
    for each_class in output_class:
        folder_path = os.path.join("20_newsgroups", str(each_class))
        for count,file_name in enumerate(os.listdir(folder_path)):
            with open(os.path.join(folder_path, str(file_name))) as f:
                val_class = output_class[each_class]
                if count%2 == 0:                 
                    train_data.append([f.read(),val_class])
                else:
                    test_data.append([f.read(),val_class])
#                    test_class.append(each_class)
#                if count == 100:
#                    break
#    print('train_data :', train_data,'| train_class :',train_class)
#    print('len(train_data): ',len(train_data),'| len(test_data): ',len(test_data))
    train_data = np.array(train_data)
#    np.random.shuffle(train_data)
    test_data = np.array(test_data)
#    print(test_data)
    return train_data,test_data,output_class

def median_words_per_sample(data):
    num_words = [len(data_sample[:][0].split()) for data_sample in data]
    return np.median(num_words)
    
def collecting_key_metrics(data,output_class):
    num_samples = data.shape[0]
    train_class = data[:,1]
    train_class = np.array(train_class)
    unique_classes, samples_per_class =np.unique(train_class,return_counts = True)
    median_of_words = median_words_per_sample(data)
    print('num_samples :',num_samples)
    print('unique_classes: ',unique_classes,'| samples_per_class: ',samples_per_class)
    print('median_of_words: ',median_of_words)
    print('num_samples/num_of_words_per_sample',num_samples//median_of_words)
    
def vectorizing(train_data,train_y,test_data):
    #    Tokenize text samples into word uni+bigrams,
    #    Vectorize using tf-idf encoding,
    #    Select only the top 20,000 features from the vector of tokens by discarding tokens that appear fewer than 2 times and using f_classif to calculate feature importance.
    
    # Range (inclusive) of n-gram sizes for tokenizing text.
    NGRAM_RANGE = (1, 2)

    # Limit on the number of features. We use the top 20K features.
    TOP_K = 20000
    # Whether text should be split into word or character n-grams.
    # One of 'word', 'char'.
    TOKEN_MODE = 'word'

    # Minimum document/corpus frequency below which a token will be discarded.
    MIN_DOCUMENT_FREQUENCY = 1
    
    keyword_args = {'ngram_range':NGRAM_RANGE,
                    'dtype':'int32',
                    'strip_accents': 'unicode',
                    'decode_error': 'replace',
                    'analyzer': TOKEN_MODE,  # Split text into word tokens.
                    'min_df': MIN_DOCUMENT_FREQUENCY
                    }
    vectorizer = TfidfVectorizer(**keyword_args)
    
    # new train_data
    train_vector = vectorizer.fit_transform(train_data)
    test_vector = vectorizer.transform(test_data)
    selector = SelectKBest(f_classif, k=min(TOP_K, train_vector.shape[1]))
    selector.fit(train_vector, train_y)
    train_vector = selector.transform(train_vector).astype('float32')
    test_vector = selector.transform(test_vector).astype('float32')
    return train_vector, test_vector

def fit_multinomialNB(train_vector,y,output_class):
#    train_vector.toarray()
#    print(train_vector)
    train_dense_vector = train_vector.todense()
    train_dense_vector = np.array(train_dense_vector)
#    print('\ntrain_dense_vector.shape: ',train_dense_vector.shape)
#    print('\ntrain_dense_vector: ',train_dense_vector)
    
    prob_f_given_c = np.zeros((len(output_class),train_dense_vector.shape[1]))
#    print('\nprob_f_given_c.shape: ',prob_f_given_c.shape)
    one_arr = np.ones((prob_f_given_c.shape[1]))
    for class_name,class_index in output_class.items():
#        print('\nBefore adding data_row to prob_f_given_c[class_index,:] \n',prob_f_given_c[class_index,:])
        for data_row,op in zip(train_dense_vector,y):
            if int(class_index) == int(op):
                prob_f_given_c[class_index,:] += data_row
#        print('\nAfter adding data_row to prob_f_given_c[class_index,:] \n',prob_f_given_c[class_index,:])
        
#        print('\nBefore adding 1 to prob_f_given_c[class_index,:] \n',prob_f_given_c[class_index,:])
        prob_f_given_c[class_index,:] += one_arr
#        print('\nAfter adding 1 to prob_f_given_c[class_index,:] \n',prob_f_given_c[class_index,:])
        
        prob_f_given_c[class_index,:] /= (((prob_f_given_c[class_index,:])-one_arr) + prob_f_given_c.shape[1])
        
#    print(prob_f_given_c)
#    mult_prob_f_given_c = np.zeros(prob_f_given_c.shape[0])
#    for i in range(mult_prob_f_given_c.shape[0]):
#        mult_prob_f_given_c[i] = np.exp(np.log(np.sum(prob_f_given_c[i,:])))
#    print('\nmult_prob_f_given_c :',mult_prob_f_given_c)
    return prob_f_given_c
    
def predict_multinomialNB(test_vector,prob_f_given_c,test_data):
    test_dense_vector = test_vector.todense()
    test_dense_vector = np.array(test_dense_vector)
    accuracy_dict = {'t':0,'f':0}
    predicted = np.zeros(test_data.shape[0])
    for index, test_point in enumerate(test_dense_vector):
        result = test_point * prob_f_given_c
#        print('\nresult :',result)
#        print('\nresult.shape : ',result.shape)
        mult_prob_f_given_c = np.zeros(prob_f_given_c.shape[0])
        for i in range(mult_prob_f_given_c.shape[0]):
            mult_prob_f_given_c[i] = np.exp(np.log(np.sum(result[i,:])))
#        print('\nmult_prob_f_given_c :',mult_prob_f_given_c*.05)
        predicted[index] = np.argmax(mult_prob_f_given_c*0.05)
#        print('\npredict class in int :',int(predicted[index]),' | test class:',test_data[index,1])
        if int(predicted[index]) == int(test_data[index,1]):
            accuracy_dict['t'] += 1
        else:accuracy_dict['f'] += 1
    print('Predictions: \n',predicted)
    print('\naccuracy_dict:', accuracy_dict)
    print('\n Accuracy :',accuracy_dict['t']*100/(accuracy_dict['t']+accuracy_dict['f']),'%')

def sklearn_multinomialNB(train_vector,train_data,test_vector,test_data):
    from sklearn.naive_bayes import GaussianNB,MultinomialNB
    clf = MultinomialNB().fit(train_vector, train_data[:,1])
    predicted = clf.predict(test_vector)
#    print('\npredicted: \n',predicted) 
    count = 0
    accuracy_dict = {'t':0,'f':0}
    while(count<predicted.shape[0]):
        if predicted[count] == str(test_data[count,1]):
            accuracy_dict['t'] += 1
        else:accuracy_dict['f'] += 1
        count += 1
    print('Predictions: \n',predicted)
    print('\naccuracy_dict:', accuracy_dict)
    print('\nAccuracy :',accuracy_dict['t']*100/(accuracy_dict['t']+accuracy_dict['f']),'%')

if __name__ == '__main__':
    print('Loading data for training and testing...')
    print('The whole program takes approx 5 mins to run, have patience...')
    train_data, test_data, output_class = load_text_data()
    print('\nLoading complete...')
    print('\nCollecting Key Metrics...')
    collecting_key_metrics(train_data,output_class)
    print('\nCollecting Done.')
    print('\nVectorizing using N-gram vectors method...')
    train_vector, test_vector = vectorizing(train_data[:,0],train_data[:,1],test_data[:,0])
    print('\nVectorization Done')

    print('\nImplementing MultinomialNaiveBayes (made from scratch)...')
    prob_f_given_c = fit_multinomialNB(train_vector,train_data[:,1],output_class)
    predict_multinomialNB(test_vector,prob_f_given_c,test_data)
    print('\nImplementing sklearn MultinomialNaiveBayes...')
    sklearn_multinomialNB(train_vector,train_data,test_vector,test_data)

###################### GaussianNB unused code ########################################
#def fit_gaussianNB(train_vector,y,output_class):
#    train_dense_vector = train_vector.todense()
#    train_dense_vector = np.array(train_dense_vector)
#    mean_train = np.zeros((len(output_class),train_dense_vector.shape[1]))
#    var_train = np.zeros((len(output_class),train_dense_vector.shape[1]))
#    print('\nmean_train.shape: ',mean_train.shape)
#    one_arr = np.ones((mean_train.shape[1]))
#    for class_name,class_index in output_class.items():
##        print('\nBefore adding data_row to mean_train[class_index,:] \n',mean_train[class_index,:])
#        count = 0
#        for data_row,op in zip(train_dense_vector,y):
#            if int(class_index) == int(op):
#                mean_train[class_index,:] += data_row
#                count += 1
##        print('\nAfter adding data_row to mean_train[class_index,:] \n',mean_train[class_index,:])
#        
##        print('\nBefore dividing COUNT to mean_train[class_index,:] \n',mean_train[class_index,:])
#        mean_train[class_index,:] /= count
##        print('\nAfter dividing COUNT to mean_train[class_index,:] \n',mean_train[class_index,:])
#        
#        count = 0
#        for data_row,op in zip(train_dense_vector,y):
#            if int(class_index) == int(op):
#                var_train[class_index,:] += np.square(data_row - mean_train[class_index,:])
#                count += 1
##        print('\nAfter adding data_row to var_train[class_index,:] \n',var_train[class_index,:])
#        
##        print('\nBefore dividing COUNT to var_train[class_index,:] \n',var_train[class_index,:])
#        var_train[class_index,:] /= (count-1)
##        print('\nAfter dividing COUNT to var_train[class_index,:] \n',var_train[class_index,:])
#    print('\n******************* MEAN ************************\n',mean_train)
#    print('\n******************* VARIANCE ************************\n',var_train)
#    return mean_train,var_train

#def predict_gaussianNB(test_vector,mean_train,var_train):
#    test_dense_vector = test_vector.todense()
#    test_dense_vector = np.array(test_dense_vector)
#    result = np.zeros((mean_train.shape[0]))
#    for test_point in test_dense_vector:
#        normal_dist_cal = np.exp(-np.square(mean_train - test_point)/(2*var_train))/(np.sqrt(var_train*2*np.pi))
#        print('\nnormal_dist_cal :',normal_dist_cal)
#        print('\nnormal_dist_cal.shape : ',normal_dist_cal.shape)
#        print(type(normal_dist_cal[0,0]))
#        
#        break
