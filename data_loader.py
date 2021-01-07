import json
import tensorflow as tf
import random
import jieba
from utils import tools as utl


def load_traindata():
    print('Train_data Processing...')

    with open('./Data/FM-CH-QA.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)

    json_data = json_data['train']

    imgId = []
    text_question = []
    text_answer = []
    max_que = 18
    max_ans = 13
    for i in json_data:
        que_list = list(jieba.cut(utl.cat_to_chs(i['Question'])))
        ans_list = list(jieba.cut(utl.cat_to_chs(i['Answer'][:-1])))
        if len(que_list) <= max_que and len(ans_list) <= max_ans:
            text_question.append('<BOA> ' + ' '.join(que_list) + ' <EOA>')
            text_answer.append('<BOA> ' + ' '.join(ans_list) + ' <EOA>')
            imgId.append(i['image_id'])

    tokenizer_que = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_que.fit_on_texts(text_question)
    tokenizer_ans = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_ans.fit_on_texts(text_answer)

    que_filter = []
    for s in tokenizer_que.word_counts:
        if tokenizer_que.word_counts[s] > 2:
            que_filter.append(s)
    print('que_vocab_len:', len(que_filter))

    ans_filter = []
    for s in tokenizer_ans.word_counts:
        if tokenizer_ans.word_counts[s] > 5:
            ans_filter.append(s)
    print('ans_vocab_len:', len(ans_filter))

    imgId = []
    text_question = []
    text_answer = []
    max_que = 18
    max_ans = 13
    for i in json_data:
        que_list = list(jieba.cut(utl.cat_to_chs(i['Question'])))
        ans_list = list(jieba.cut(utl.cat_to_chs(i['Answer'][:-1])))
        if len(que_list) <= max_que and len(ans_list) <= max_ans:
            for n in range(len(que_list)):
                if que_list[n] not in que_filter:
                    que_list[n] = '<unk>'
            for n in range(len(ans_list)):
                if ans_list[n] not in ans_filter:
                    ans_list[n] = '<unk>'

            text_question.append('<boa> ' + ' '.join(que_list) + ' <eoa>')
            text_answer.append('<boa> ' + ' '.join(ans_list) + ' <eoa>')
            imgId.append(i['image_id'])

    tokenizer_que = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_que.fit_on_texts(text_question)
    tokenizer_ans = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_ans.fit_on_texts(text_answer)

    seq_ans = tokenizer_ans.texts_to_sequences(text_answer)
    seq_que = tokenizer_que.texts_to_sequences(text_question)

    maxlen_que = utl.getmaxlength(seq_que)
    maxlen_ans = utl.getmaxlength(seq_ans)

    seq_que = tf.keras.preprocessing.sequence.pad_sequences(seq_que, padding='post')
    seq_ans = tf.keras.preprocessing.sequence.pad_sequences(seq_ans, padding='post')

    dataset = tf.data.Dataset.from_tensor_slices({
        'id': imgId,
        'question': seq_que,
        'answer': seq_ans,
    })

    print('Train_data OK!')

    return dataset, tokenizer_que, tokenizer_ans, max_que, max_ans


def load_valdata(tokenizer_que, tokenizer_ans):
    print('Val_data Processing...')
    data_num = 12800

    with open('./Data/FM-CH-QA.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)

    json_data = json_data['val']

    imgId = []
    text_question = []
    text_answer = []
    max_que = 18
    max_ans = 13
    for i in json_data:
        que_list = list(jieba.cut(utl.cat_to_chs(i['Question'])))
        ans_list = list(jieba.cut(utl.cat_to_chs(i['Answer'][:-1])))
        if len(que_list) <= max_que and len(ans_list) <= max_ans:
            text_question.append('<boa> ' + ' '.join(que_list) + ' <eoa>')
            text_answer.append('<boa> ' + ' '.join(ans_list) + ' <eoa>')
            imgId.append(i['image_id'])

    random.seed(3)
    random.shuffle(imgId)
    random.seed(3)
    random.shuffle(text_question)
    random.seed(3)
    random.shuffle(text_answer)
    imgId = imgId[:data_num]
    text_question = text_question[:data_num]
    text_answer = text_answer[:data_num]

    que_vocab = list(tokenizer_que.word_counts)
    ans_vocab = list(tokenizer_ans.word_counts)

    for i, sen in enumerate(text_question):
        sen = sen.split(' ')
        for j in range(len(sen)):
            if sen[j] not in que_vocab:
                sen[j] = '<unk>'
        text_question[i] = sen

    for i, sen in enumerate(text_answer):
        sen = sen.split(' ')
        for j in range(len(sen)):
            if sen[j] not in ans_vocab:
                sen[j] = '<unk>'
        text_answer[i] = sen

    seq_ans = tokenizer_ans.texts_to_sequences(text_answer)
    seq_que = tokenizer_que.texts_to_sequences(text_question)

    maxlen_que = utl.getmaxlength(seq_que)
    maxlen_ans = utl.getmaxlength(seq_ans)

    seq_que = tf.keras.preprocessing.sequence.pad_sequences(seq_que, padding='post')
    seq_ans = tf.keras.preprocessing.sequence.pad_sequences(seq_ans, padding='post')

    dataset = tf.data.Dataset.from_tensor_slices({
        'id': imgId,
        'question': seq_que,
        'answer': seq_ans,
    })
    print('Val_data OK!')

    return dataset, maxlen_que, maxlen_ans


