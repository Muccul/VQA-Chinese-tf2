import os
import numpy as np
import tensorflow as tf
from PIL import Image
from utils.langconv import Converter
from nltk.translate.bleu_score import sentence_bleu


def cat_to_chs(sentence):  # 传入参数为列表
    """
    将繁体转换成简体
    :param line:
    :return:
    """
    sentence = Converter('zh-hans').convert(sentence)
    sentence.encode('utf-8')
    return sentence


def convert_text(lang_tokenizer, tensor):
    text = ''
    no_print = [0, lang_tokenizer.word_index['<boa>'], lang_tokenizer.word_index['<eoa>']]
    for t in tensor:
        if t not in no_print:
            text += lang_tokenizer.index_word[t.numpy()]
            text += ' '
    return text


def getpath(img_id, mode='train'):
    root_file = {
        'train': r'./Data/train2014/',
        'val': r'./Data/val2014/'
    }
    root = root_file[mode]

    if mode == 'train':
        return os.path.join(root, 'COCO_train2014_' + (img_id.rjust(12, '0')) + '.jpg')
    else:
        return os.path.join(root, 'COCO_val2014_' + (img_id.rjust(12, '0')) + '.jpg')


def getmaxlength(seq):
    maxlen = 0
    for i in seq:
        if len(i) > maxlen :
            maxlen = len(i)
    return maxlen


def get_bleu(reference, candidate, smooth):
    score = sentence_bleu([reference], candidate,
                          (0.5, 0.5),
                          smoothing_function=smooth.method1)
    return score


def get_rouge1(reference, candidate):
    grams_reference = reference
    grams_model = candidate
    temp = 0
    ngram_all = len(grams_reference)
    for x in grams_reference:
        if x in grams_model: temp = temp + 1
    rouge_1 = temp / ngram_all
    return rouge_1


def get_rouge2(reference, candidate):
    grams_reference = reference
    grams_model = candidate
    gram_2_model = []
    gram_2_reference = []
    temp = 0
    ngram_all = len(grams_reference) - 1
    for x in range(len(grams_model) - 1):
        gram_2_model.append(grams_model[x] + grams_model[x + 1])
    for x in range(len(grams_reference) - 1):
        gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])
    for x in gram_2_model:
        if x in gram_2_reference: temp = temp + 1
    rouge_2 = temp / ngram_all
    return rouge_2


def get_img_tensor(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = (img / 255.0).astype('float32')
    img = img[tf.newaxis, ...]
    return img
