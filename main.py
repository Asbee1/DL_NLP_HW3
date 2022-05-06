import jieba, os, re
import numpy as np
from gensim import corpora, models
from operator import itemgetter


def get_data():

    outfilename_1 = "./train.txt" #用于训练的文本
    outfilename_2 = "./test.txt"  #用于测试的文本
    #每个人选择的训练和测试的文本不一样 导致最后的结果肯定不一样

    if not os.path.exists('./train.txt'):
        outputs = open(outfilename_1, 'w', encoding='UTF-8')
        outputs_test = open(outfilename_2, 'w', encoding='UTF-8')
        datasets_root = "./datasets1"
        catalog = "inf.txt"

        test_num = 10*50 #每本测试的数量
        with open(os.path.join(datasets_root, catalog), "r", encoding='utf-8') as f:#打开目录文件
            all_files = f.readline().split(",")#把每一个小说名字分开
            print(all_files)

        for name in all_files: #对于每一个名字
            with open(os.path.join(datasets_root, name + ".txt"), "r", encoding='utf-8') as f:#打开每一个名字对应的小说文本
                file_read = f.readlines() #将小说所有行都保存起来
                train_num = len(file_read) - test_num  #除掉测试行数(在每一个小说里面选择多少行文本)的所有都用于训练
                choice_index = np.random.choice(len(file_read), test_num + train_num, replace=False)
                train_text = ""
                for train in choice_index[0:train_num]: #顺序打乱之后 取前train_num行
                    line = file_read[train]
                    line = re.sub('\s', '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    if len(line) == 0:
                        continue
                    seg_list = list(jieba.cut(line, cut_all=False))  # 使用精确模式  进行分词
                    line_seg = ""
                    for term in seg_list: #对于刚才选取的分词后的该每行文本
                        line_seg += term + " " #将他们保存为每一个词+一个空格的形式 直接储存在line_seg中
                        # for index in range len(line_seg):
                    outputs.write(line_seg.strip() + '\n') #输出到训练集中

                for px in range(10):
                    for test in choice_index[train_num + (px-1)*50:test_num + train_num]:#最后的段落是测试集

                        test_line = ""
                    # for i in range(test, test + test_length):
                        line = file_read[test]
                        line = re.sub('\s', '', line)
                        line = re.sub('\n', '', line)
                        line = re.sub('[\u0000-\u4DFF]', '', line)
                        line = re.sub('[\u9FA6-\uFFFF]', '', line)
                        seg_list = list(jieba.cut(line, cut_all=False))  # 使用精确模式 进行分词
                        #line_seg = ""
                        for term in seg_list:
                            test_line += term + " "#分完词之后 把这一行变成词+空格的形式
                        outputs_test.write(test_line.strip())
                    outputs_test.write('\n')
                outputs_test.write('\n')

        outputs.close()
        outputs_test.close()
        print("得到训练和测试文本")

if __name__ == "__main__":
    get_data()

    #得到训练集和测试集了

    fr = open('./train.txt', 'r', encoding='utf-8')
    train = []
    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ')]#对空格进行分字符
        train.append(line)#把该line中所有单词变成一个list

    """训练LDA模型"""
    dictionary = corpora.Dictionary(train)
    # corpus[0]: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),...]
    # corpus是把每本小说ID化后的结果，每个元素是新闻中的每个词语，在字典中的ID和频率
    corpus = [dictionary.doc2bow(text) for text in train]
    #
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=16)
    # lsi = models.LsiModel(corpus=corpus,id2word=dictionary,num_topics=16)

    #对lda model的结果进行输出

    top_topics = lda.top_topics(corpus) #, num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / 16
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    from pprint import pprint
    pprint(top_topics)
    # 输出结束

    topic_list_lda = lda.print_topics(16)
    # topic_list_lsi = lsi.print_topics(16)
    print("16个主题的单词分布为：\n")
    for topic in topic_list_lda:
        print(topic)


    file_test = "./test.txt"
    news_test = open(file_test, 'r', encoding='UTF-8')
    test = []

    # 处理成正确的输入格式
    for line in news_test:
        line = [word.strip() for word in line.split(' ')]
        test.append(line)#对于该行 将该行的所有单词分开 组成一个list

    for text in test: #
        corpus_test = dictionary.doc2bow((text))#最终得到的是每段文本的词典
        # print(corpus_test)

    corpus_test = [dictionary.doc2bow(text) for text in test]
    # print(corpus_test)
    # 得到每个测试集的主题分布

    topics_test = lda.get_document_topics(corpus_test)
    data_1 = []
    data_2 = []
    for px in range(160):
        data_1.append(max(topics_test[px-1],key=itemgetter(1))[0])

    print(data_1)
    for i in range(17):
        print(f'第{i}组分布')
        print(data_1[(i-1)*10:i*10])

    for i in range(160):
        print(i)
        print('的主题分布为：\n')
        print(topics_test[i], '\n')
    fr.close()
    news_test.close()

