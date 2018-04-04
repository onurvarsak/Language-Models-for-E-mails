import numpy as np
import math
import time
import sys
import csv


start_time = time.time()


input_file_name = sys.argv[1]                            #this variable will be input filename and type
output_file_name = sys.argv[2]                           #this variable will be output filename




#this function read txt type files and return all lines like list
def read_txt_file(filename):
    file = open(filename, encoding="utf8")
    return file.readlines()

#this function takes %60 of data and this data is called "train set"
def get_train_set(data, trainSet, size):
    global data_set_token_size
    for i in range(size):
        trainSet.append(data[i])
        trainSet[i] = trainSet[i].lower().replace("\n", "")
        trainSet[i] = trainSet[i].split(" ")
        data_set_token_size = data_set_token_size + len(trainSet[i])

#this function calculates uni, bi, trigram token frequency
def get_frequency(trainSet, unigram, bigram, trigram):
    for line in trainSet:
        b_first = "<s>"

        t_first = "<s>"
        t_second = "<s>"

        for index in range(len(line)):
            if(index + 1 == len(line)):
                add_unigram_token(line[index], unigram)

                add_ngram_token(b_first, line[index], bigram)
                b_first = line[index]
                add_ngram_token(b_first, "</s>", bigram)

                add_ngram_token(t_first + " " + t_second, line[index], trigram)
                t_first = t_second
                t_second = line[index]
                add_ngram_token(t_first + " " + t_second, "</s>", trigram)
            else:
                add_unigram_token(line[index], unigram)

                add_ngram_token(b_first, line[index], bigram)
                b_first = line[index]

                add_ngram_token(t_first + " " + t_second, line[index], trigram)
                t_first = t_second
                t_second = line[index]

#this function adds token in unigram frequency data set
def add_unigram_token(token, ngram):
    if(token not in ngram.keys()):
        ngram[token] = 1
    else:
        ngram[token] = ngram[token] + 1

#this function adds token in bigram or trigram frequenct data set
def add_ngram_token(first_token, second_token, ngram):
    if(first_token in ngram.keys()):
        if(second_token in ngram[first_token][0].keys()):
            ngram[first_token][0][second_token] = ngram[first_token][0][second_token] + 1
            ngram[first_token][1] = ngram[first_token][1] + 1
        else:
            ngram[first_token][0][second_token] = 1
            ngram[first_token][1] = ngram[first_token][1] + 1
    else:
        ngram[first_token] = [{}, 0]
        ngram[first_token][0][second_token] = 1
        ngram[first_token][1] = ngram[first_token][1] + 1

#this function generates 10 nonsmooting unigram sentences
def get_nonsmooting_unigram_sentence(unigrams_token_frequency, data_set_token_size):
    out_sentences = ""

    unigram_probability_sum = 0
    for token in unigrams_token_frequency.keys():
        unigram_probability_sum = unigram_probability_sum + (int(unigrams_token_frequency[token]) / data_set_token_size)


    sentence_probability = 1
    sentence_token_size = 0
    end_words = [".", "!", "?"]
    sentence = ""
    flag_sum = 0
    token = ""


    for size in range(10):
        while(1):

            if(sentence_token_size == 30 or token in end_words):
                break

            random_number = np.random.uniform(0, unigram_probability_sum, 1)[0]

            for token in unigrams_token_frequency.keys():

                token_probability = int(unigrams_token_frequency[token]) / data_set_token_size
                flag_sum = flag_sum + token_probability

                if(flag_sum >= random_number):
                    sentence_probability = sentence_probability * token_probability

                    if(sentence == ""):
                        sentence = token
                    else:
                        sentence = sentence + " " + token

                    sentence_token_size = sentence_token_size + 1
                    flag_sum = 0
                    break
        out_sentences = out_sentences + str(sentence_probability) + "\t\t" + sentence + "\n"
        #print(str(sentence_probability) + "\t\t" + sentence)
        sentence_probability = 1
        sentence_token_size = 0
        sentence = ""
        token = ""

    return out_sentences

#this function generates 10 unigram sentences with my smoothing model
def get_smoothing_unigram_sentence(unigrams_token_frequency, data_set_token_size):
    out_sentences = ""

    unigram_probability_sum = 0
    for token in unigrams_token_frequency.keys():
        unigram_probability_sum = unigram_probability_sum + ((int(unigrams_token_frequency[token]) + 1) / (data_set_token_size + len(unigrams_token_frequency)))

    sentence_probability = 1
    sentence_token_size = 0
    end_words = [".", "!", "?"]
    sentence = ""
    flag_sum = 0
    token = ""

    for size in range(10):
        while(1):

            if(sentence_token_size == 30 or token in end_words):
                break

            random_number = np.random.uniform(0, unigram_probability_sum, 1)[0]

            for token in unigrams_token_frequency.keys():

                token_probability = ((int(unigrams_token_frequency[token]) + 1) / (data_set_token_size + len(unigrams_token_frequency)))
                flag_sum = flag_sum + token_probability

                if(flag_sum >= random_number):
                    sentence_probability = sentence_probability * token_probability

                    if(sentence == ""):
                        sentence = token
                    else:
                        sentence = sentence + " " + token

                    sentence_token_size = sentence_token_size + 1
                    flag_sum = 0
                    break

        out_sentences = out_sentences + str(sentence_probability) + "\t\t" + sentence + "\n"
        #print(str(sentence_probability) + "\t\t" + sentence)
        sentence_probability = 1
        sentence_token_size = 0
        sentence = ""
        token = ""

    return out_sentences

#this function generates 10 nonsmooting bigram sentences
def get_nonsmoothing_bigram_sentence(bigrams_token_frequency):
    out_sentences = ""

    sentence_probability = 1
    sentence_token_size = 0
    first_token = "<s>"
    end_words = [".", "!", "?", "</s>"]
    flag_sum = 0
    sentence = ""


    for size in range(10):
        while(1):

            if(sentence_token_size == 30 or first_token in end_words):
                    break

            #fisrt_token daki tokenların toplam probabilitysi 1 e ulaşmadığı için toplamı buluyoruz
            sum_probability = 0
            for second_token in bigrams_token_frequency[first_token][0].keys():
                sum_probability = sum_probability + (int(bigrams_token_frequency[first_token][0][second_token]) / bigrams_token_frequency[first_token][1])

            random_number = np.random.uniform(0, sum_probability, 1)[0]

            for second_token in bigrams_token_frequency[first_token][0].keys():

                token_probability = bigrams_token_frequency[first_token][0][second_token] / bigrams_token_frequency[first_token][1]
                flag_sum = flag_sum + token_probability

                if(flag_sum >= random_number):

                    if(second_token == "</s>"):
                        first_token = second_token
                        flag_sum = 0
                        break

                    sentence_probability = sentence_probability * token_probability
                    first_token = second_token

                    if(sentence == ""):
                        sentence = second_token
                    else:
                        sentence = sentence + " " + second_token
                    sentence_token_size = sentence_token_size + 1
                    flag_sum = 0
                    break

        out_sentences = out_sentences + str(sentence_probability) + "\t\t" + sentence + "\n"
        #print(str(sentence_probability) + "\t\t" + sentence)
        sentence_probability = 1
        sentence_token_size = 0
        first_token = "<s>"
        sentence = ""

    return out_sentences

#this function generates 10 bigram sentences with my smoothing model
def get_smoothing_bigram_sentence(unigram_token_frequency, bigrams_token_frequency):
    out_sentences = ""

    sentence_probability = 1
    sentence_token_size = 0
    first_token = "<s>"
    end_words = [".", "!", "?",  "</s>"]
    flag_sum = 0
    sentence = ""


    for size in range(10):
        while(1):

            if(sentence_token_size == 30 or first_token in end_words):
                    break

            #fisrt_token daki tokenların toplam probabilitysi 1 e ulaşmadığı için toplamı buluyoruz
            sum_probability = 0
            for second_token in bigrams_token_frequency[first_token][0].keys():
                sum_probability = sum_probability + ((int(bigrams_token_frequency[first_token][0][second_token]) + 1) / (bigrams_token_frequency[first_token][1] + len(unigram_token_frequency)))

            random_number = np.random.uniform(0, sum_probability, 1)[0]

            for second_token in bigrams_token_frequency[first_token][0].keys():

                token_probability = ((int(bigrams_token_frequency[first_token][0][second_token]) + 1) / (bigrams_token_frequency[first_token][1] + len(unigram_token_frequency)))
                flag_sum = flag_sum + token_probability

                if(flag_sum >= random_number):

                    if(second_token == "</s>"):
                        first_token = second_token
                        flag_sum = 0
                        break

                    sentence_probability = sentence_probability * token_probability
                    first_token = second_token

                    if(sentence == ""):
                        sentence = second_token
                    else:
                        sentence = sentence + " " + second_token

                    sentence_token_size = sentence_token_size + 1
                    flag_sum = 0
                    break
        out_sentences = out_sentences + str(sentence_probability) + "\t\t" + sentence + "\n"
        #print(str(sentence_probability) + "\t\t" + sentence)
        sentence_probability = 1
        sentence_token_size = 0
        first_token = "<s>"
        sentence = ""

    return out_sentences

#this function generates 10 nonsmooting trigram sentences
def get_nonsmoothing_trigram_sentence(trigrams_token_frequency):
    out_sentences = ""

    sentence_probability = 1
    sentence_token_size = 0
    second_token = "<s>"
    first_token = "<s>"
    end_words = [".", "!", "?", "</s>"]
    flag_sum = 0
    sentence = ""

    for size in range(10):

        while(1):

            if(sentence_token_size == 30 or second_token in end_words):
                break

            sum_probability = 0
            for third_token in trigrams_token_frequency[first_token + " " + second_token][0].keys():
                sum_probability = sum_probability + (int(trigrams_token_frequency[first_token + " " + second_token][0][third_token]) / trigrams_token_frequency[first_token + " " + second_token][1])

            random_number = np.random.uniform(0, sum_probability, 1)[0]

            for third_token in trigrams_token_frequency[first_token + " " + second_token][0].keys():
                token_probability = trigrams_token_frequency[first_token + " " + second_token][0][third_token] / trigrams_token_frequency[first_token + " " + second_token][1]
                flag_sum = flag_sum + token_probability

                if(flag_sum >= random_number):

                    if(third_token == "</s>"):
                        second_token = third_token
                        flag_sum = 0
                        break

                    sentence_probability = sentence_probability * token_probability
                    first_token = second_token
                    second_token = third_token

                    if(sentence == ""):
                        sentence = third_token
                    else:
                        sentence = sentence + " " + third_token

                    sentence_token_size = sentence_token_size + 1
                    flag_sum = 0
                    break
        out_sentences = out_sentences + str(sentence_probability) + "\t\t" + sentence + "\n"
        #print(str(sentence_probability) + "\t\t" + sentence)
        sentence_probability = 1
        sentence_token_size = 0
        second_token = "<s>"
        first_token = "<s>"
        sentence = ""

    return out_sentences

#this function generates 10 trigram sentences with my smoothing model
def get_smoothing_trigram_sentence(bigram_token_frequency, trigrams_token_frequency):
    out_sentences = ""

    sentence_probability = 1
    sentence_token_size = 0
    second_token = "<s>"
    first_token = "<s>"
    end_words = [".", "!", "?", "</s>"]
    flag_sum = 0
    sentence = ""

    for size in range(10):

        while(1):

            if(sentence_token_size == 30 or second_token in end_words):
                break

            sum_probability = 0
            for third_token in trigrams_token_frequency[first_token + " " + second_token][0].keys():
                sum_probability = sum_probability + ((int(trigrams_token_frequency[first_token + " " + second_token][0][third_token]) + 1) / (trigrams_token_frequency[first_token + " " + second_token][1] + len(bigram_token_frequency)))

            random_number = np.random.uniform(0, sum_probability, 1)[0]

            for third_token in trigrams_token_frequency[first_token + " " + second_token][0].keys():
                token_probability = ((int(trigrams_token_frequency[first_token + " " + second_token][0][third_token]) + 1) / (trigrams_token_frequency[first_token + " " + second_token][1] + len(bigram_token_frequency)))
                flag_sum = flag_sum + token_probability

                if(flag_sum >= random_number):
                    if(third_token == "</s>"):
                        second_token = third_token
                        flag_sum = 0
                        break

                    sentence_probability = sentence_probability * token_probability
                    first_token = second_token
                    second_token = third_token

                    if(sentence == ""):
                        sentence = third_token
                    else:
                        sentence = sentence + " " + third_token

                    sentence_token_size = sentence_token_size + 1
                    flag_sum = 0
                    break
        out_sentences = out_sentences + str(sentence_probability) + "\t\t" + sentence + "\n"
        #print(str(sentence_probability) + "\t\t" + sentence)
        sentence_probability = 1
        sentence_token_size = 0
        second_token = "<s>"
        first_token = "<s>"
        sentence = ""

    return out_sentences


#this task generate sentences that nonsmoothing and smoothing
def tast_3(uni_freq, bi_freq, tri_freq, data_set_token_size):

    out = ""

    out = out + "TASK 3 - GENERATE SENTENCE 10 TIMES EVERY MODEL\n\n\n"
    out = out + "NonSmoothing Unigram Sentences\n\n"
    out = out + get_nonsmooting_unigram_sentence(uni_freq, data_set_token_size)

    out = out + "\nSmoothing Unigram Sentences\n\n"
    out = out + get_smoothing_unigram_sentence(uni_freq, data_set_token_size)

    out = out + "\nNonSmoothing Bigram Sentences\n\n"
    out = out + get_nonsmoothing_bigram_sentence(bi_freq)

    out = out + "\nSmoothing Bigram Sentences\n\n"
    out = out + get_smoothing_bigram_sentence(uni_freq, bi_freq)

    out = out + "\nNonSmoothing Trigram Sentences\n\n"
    out = out + get_nonsmoothing_trigram_sentence(tri_freq)

    out = out + "\nSmoothing Trigram Sentences\n\n"
    out = out + get_smoothing_trigram_sentence(bi_freq, tri_freq)

    readFile(out)

#this does task2 and task4
def calculate_probability_and_perplexity(data, start, end, big_freq, tri_freq, uni_freq):
    readFile("\n\n\n\n\nTASK 2 and TASK 4 -- PROBABILITY and PERPLEXITY\n\n\n")

    testSet = []
    index = 0
    for i in range(start, end):
        testSet.append("<s> <s> " + data[i] + " </s>")
        testSet[index] = testSet[index].lower().replace("\n", "")
        testSet[index] = testSet[index].split(" ")

        sentence_probability = 1

        first_token = ""
        second_token = testSet[index][0]
        third_token = testSet[index][1]

        big_perp = 0
        tri_perp = 0

        #probability hesaplama
        for j in range(2, len(testSet[index])):
            first_token = second_token
            second_token = third_token
            third_token = testSet[index][j]

            if(first_token + " " + second_token in tri_freq.keys()):
                if(third_token in tri_freq[first_token + " " + second_token][0].keys()):
                    tri_result = (tri_freq[first_token + " " + second_token][0][third_token] + 1) / (tri_freq[first_token + " " + second_token][1] + len(uni_freq))
                    big_result = (big_freq[second_token][0][third_token] + 1) / (big_freq[second_token][1] + len(uni_freq))
                    sentence_probability = sentence_probability + math.log( ((tri_freq[first_token + " " + second_token][0][third_token] + 1) / (tri_freq[first_token + " " + second_token][1] + len(big_freq))),2)
                else:
                    tri_result = 1 / (tri_freq[first_token + " " + second_token][1] + len(uni_freq))
                    big_result = 1 / (big_freq[second_token][1] + len(uni_freq))
                    sentence_probability = sentence_probability + math.log((1 / (tri_freq[first_token + " " + second_token][1] + len(big_freq))),2)
            else:
                tri_result = 1 / (1 + len(uni_freq))
                big_result = 1 / (1 + len(uni_freq))
                sentence_probability = sentence_probability + math.log((1 / (1 + len(big_freq))),2)

            tri_perp = tri_perp + math.log(tri_result,2)
            big_perp = big_perp + math.log(big_result,2)

        #sentence_probability = int(pow(2, sentence_probability))
        tri_perp = int(pow(2, (float(-1 / len(testSet[index])) * tri_perp)))
        big_perp = int(pow(2, (float(-1 / len(testSet[index])) * big_perp)))

        readFile("\nSentence:\t\t" + data[i].replace("\n", "") + "\n\nLoG Probability:\t\t\t" + str(sentence_probability) + "\nPerplexity Smoothing Bigram:\t\t" + str(big_perp) + "\nPerplexity Smoothing Trigram:\t\t" + str(tri_perp) + "\n\n")

        index = index + 1

#this function write every line to output file
def readFile(sentence):
    global output_file_name
    file = open(output_file_name, "a")
    file.write(sentence)
    file.close()


def get_message_parsing(email):
    replace_dict = {"_":" _ ", "\n": "", ".":" . ", ",":" , ", "(":" ( ", ")":" ) ", "'":" '", ":":" : ", "!":" ! ", "-":" - ","pm":" pm ", "am":" am ", ">":" > ", "<":" < " }

    start = email.find("X-FileName: ")
    end = 0
    for i in range(start, len(email)):
        if(email[i] == "\n"):
            end = i
            break

    email = email[end:]

    for i, j in replace_dict.items():
        email = email.replace(i, j)
    email = email.lower() + "\n"
    return email


#this function read csv type files and return all lines like list
def read_csv_file(filename, data):

    file = open(filename)
    file = csv.DictReader(file)
    index = 0
    for row in file:

        if (index == 15000):
            break

        index = index + 1
        data.append(get_message_parsing(row["message"]))


# this function takes %60 of data and this data is called "train set"
def get_train_set_csv(data, trainSet, size):
    global data_set_token_size

    end_token = [".", "?", "!"]

    for i in range(size):

        data[i] = data[i].lower().replace("\n", "")
        data[i] = data[i].split(" ")

        sentence_list = []

        for j in range(len(data[i])):
            if(data[i][j] == ""):
                if(j == len(data[i]) - 1 and len(sentence_list) != 0):
                    trainSet.append(sentence_list)
                    sentence_list = []
                else:
                    continue
            elif(data[i][j] in end_token):
                sentence_list.append(data[i][j])
                data_set_token_size = data_set_token_size + 1
                trainSet.append(sentence_list)
                sentence_list = []
            else:
                sentence_list.append(data[i][j])
                data_set_token_size = data_set_token_size + 1




data = []
train_set= []

train_unigram_data = {}
train_bigram_data = {}
train_trigram_data = {}


data_set_token_size = 0                                 # this variable takes token size of all train set







if(input_file_name.split(".")[1] == "txt"):
    data = read_txt_file(input_file_name)               # this instruction read and take all lines of data

    train_size = math.floor(int(len(data) * 0.6))       # this variable takes size of %60 of data lines
    test_size = math.ceil(int(len(data) * 0.4))         # this variable takes size of %40 of data lines

    # this calculate is 5 seconds approximately
    get_train_set(data, train_set, train_size)          #this instruction fills train_set variable


elif(input_file_name.split(".")[1] == "csv"):
    read_csv_file(input_file_name, data)

    train_size = math.floor(int(len(data) * 0.6))
    test_size = math.ceil(int(len(data) * 0.4))


    get_train_set_csv(data, train_set, train_size)


get_frequency(train_set, train_unigram_data, train_bigram_data, train_trigram_data)

tast_3(train_unigram_data, train_bigram_data, train_trigram_data, data_set_token_size)

calculate_probability_and_perplexity(data, train_size, train_size + test_size, train_bigram_data,
                                         train_trigram_data, train_unigram_data)

