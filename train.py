import tensorflow as tf
import os
import random
import jieba
import numpy as np
import h5py
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import utils.tools as utl
from data_loader import load_traindata, load_valdata
from vis_lstm_san import VisLSTM

EPOCHS = 15
BATCH_SIZE = 64
EMB_SIZE = 512
HIDDEN_SIZE = 512
ATT_STEPS = 2
train_h5_name = r'C:/train_block5.h5'
val_h5_name = r'C:/val_block5.h5'
checkpoint_dir = './vislstm_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

def main():
    config_gpu()
    h5f_train = h5py.File(train_h5_name, 'r')
    h5f_val = h5py.File(val_h5_name, 'r')
    dataset_train_ori, tokenizer_que, tokenizer_ans, max_que_train, max_ans_train = load_traindata()
    dataset_val_ori, max_que_val, max_ans_val = load_valdata(tokenizer_que, tokenizer_ans)
    vocab_size_que = len(tokenizer_que.word_counts)
    vocab_size_ans = len(tokenizer_ans.word_counts)
    ans_eoa_index = tokenizer_ans.word_index['<eoa>']
    ans_boa_index = tokenizer_ans.word_index['<boa>']

    dataset_train = dataset_train_ori.shuffle(160000).batch(BATCH_SIZE, drop_remainder=True)
    dataset_val = dataset_val_ori.batch(BATCH_SIZE, drop_remainder=True)

    model = VisLSTM(emb_dim=EMB_SIZE, hidden_dim=HIDDEN_SIZE,
                    vocab_size_que=vocab_size_que, vocab_size_ans=vocab_size_ans, att_steps=ATT_STEPS)
    optimizer = tf.keras.optimizers.Adam(clipnorm=1)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='categorical_crossentropy',
                                                              reduction='none')

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    latest_file = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(latest_file)

    now_epoch = 1
    if latest_file is not None:
        now_epoch = int(latest_file.split('-')[-1]) + 1

    for epoch in range(now_epoch, EPOCHS+1):
        total_train_loss = 0
        steps_train = 0
        for data in dataset_train:
            loss = train(model, data, h5f_train, optimizer, criterion)
            total_train_loss += loss
            steps_train += 1
            if steps_train % 500 == 0 and steps_train != 0:
                print(f'Epoch{epoch} Batch {steps_train} train_loss: {total_train_loss/steps_train}')

        checkpoint.save(file_prefix=checkpoint_prefix)
        print('*' * 50)
        bleu, rouge1, rouge2 = evaluate_meteric(model, dataset_val, ans_boa_index, ans_eoa_index, h5f_val)
        print(f'Epoch{epoch} train_loss: {total_train_loss/steps_train}')
        print(f'Epoch{epoch} val :{bleu},{rouge1},{rouge2}')
        print('*' * 50)

    h5f_train.close()
    h5f_val.close()

def config_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第一块GPU可用

        except RuntimeError as e:
            print(e)

def train(model, train_data, h5f_train, optimizer, criterion):
    with tf.GradientTape() as tape:
        src = train_data['question']
        trg = train_data['answer']
        imgs_features = []
        for n in range(BATCH_SIZE):
            img_tmp = np.array(h5f_train[train_data['id'][n].numpy().decode('utf-8')], dtype=np.float32)
            img_tmp = tf.reshape(img_tmp, (-1, 512))
            imgs_features.append(img_tmp)
        imgs_features = tf.stack(imgs_features, 0)

        out, _ = model(src, trg, imgs_features)

        trg = tf.reshape(trg[:, 1:], (-1,))
        out = tf.reshape(out, (-1, out.shape[-1]))

        mask = tf.math.logical_not(tf.math.equal(trg, 0))
        loss = criterion(trg, out)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_mean(loss)

    variables = model.encoder.trainable_variables + model.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


def evaluate(model, dataset_val, h5f_val, criterion):
    total_loss_val = 0
    steps_per_epoch = 0
    for data in dataset_val:
        src = data['question']
        trg = data['answer']
        imgs_features = []
        for n in range(BATCH_SIZE):
            img_tmp = np.array(h5f_val[data['id'][n].numpy().decode('utf-8')], dtype=np.float32)
            img_tmp = tf.reshape(img_tmp, (-1, 512))
            imgs_features.append(img_tmp)
        imgs_features = tf.stack(imgs_features, 0)

        out, _ = model(src, trg, imgs_features)

        trg = tf.reshape(trg[:, 1:], (-1,))
        out = tf.reshape(out, (-1, out.shape[-1]))

        mask = tf.math.logical_not(tf.math.equal(trg, 0))
        loss = criterion(trg, out)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_mean(loss)

        total_loss_val += loss
        steps_per_epoch += 1
    return total_loss_val / steps_per_epoch


def evaluate_meteric(model, dataset, ans_boa_index, ans_eoa_index, h5f_data):
    smooth = SmoothingFunction()
    total_bleu, total_rouge1, total_rouge2 = 0, 0, 0
    num = 0
    for data in dataset:
        src = data['question']
        trg = data['answer']
        imgs_features = []
        for n in range(BATCH_SIZE):
            img_tmp = np.array(h5f_data[data['id'][n].numpy().decode('utf-8')], dtype=np.float32)
            img_tmp = tf.reshape(img_tmp, (-1, 512))
            imgs_features.append(img_tmp)
        imgs_features = tf.stack(imgs_features, 0)

        out, _ = model(src, trg, imgs_features)

        predict = tf.argmax(out, axis=-1)
        for n in range(BATCH_SIZE):
            refernce = []
            candidate = []
            # processing reference
            for i in trg[n]:
                refernce.append(int(i))
                if i == ans_eoa_index or i == 0:
                    break
                    # processing candidate
            candidate.append(ans_boa_index)
            for i in predict[n]:
                candidate.append(int(i))
                if i == ans_eoa_index or i == 0:
                    break
            bleu = utl.get_bleu(refernce, candidate, smooth)
            rouge1 = utl.get_rouge1(refernce, candidate)
            rouge2 = utl.get_rouge2(refernce, candidate)
            total_bleu += bleu
            total_rouge1 += rouge1
            total_rouge2 += rouge2
            num += 1

    return total_bleu / num, total_rouge1 / num, total_rouge2 / num


def vis_test(model, data, h5f_data, tokenizer_que, tokenizer_ans, mode='val'):
    src = data['question']
    trg = data['answer']
    imgs_features = []
    for n in range(BATCH_SIZE):
        imgs_features.append(np.array(h5f_data[data['id'][n].numpy().decode('utf-8')], dtype=np.float32).squeeze())
    imgs_features = tf.stack(imgs_features, 0)

    out, _ = model(src, trg, imgs_features)

    predict = tf.argmax(out, axis=-1)
    for n in range(BATCH_SIZE):
        print(utl.getpath(data['id'][n].numpy().decode('utf-8'), mode=mode))
        print(utl.convert_text(tokenizer_que, src[n]))
        print(utl.convert_text(tokenizer_ans, trg[n]))
        print(utl.convert_text(tokenizer_ans, predict[n]))
        print('*' * 30)


if __name__ == '__main__':
    main()
