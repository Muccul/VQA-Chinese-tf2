import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

class Encoder(Model):
    def __init__(self, vocab_size, emb_dim, hid_dim, att_steps=2):
        super(Encoder, self).__init__()

        self.hid_him = hid_dim
        self.att_steps = att_steps
        self.vocab_size = vocab_size

        self.embedding = Embedding(vocab_size + 1, emb_dim)

        self.lstm = LSTM(hid_dim, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        self.emb_img = Dense(hid_dim)

        self.layer_img = Dense(hid_dim, activation='tanh')
        self.layer_que = Dense(hid_dim, activation='tanh')
        self.layer_temp = Dense(1)
        # self.dropout = Dropout(0.3)

    def call(self, x, image_feature):
        que_emb = self.embedding(x)      # x(b,t)  que_ebm(b,t,d_e)
        _, hidden_text, cell = self.lstm(que_emb)  # hidden_text(b,d_h)
        img_emb = self.emb_img(image_feature)  # img_emb(b,m,d_h)

        context = hidden_text   # context(b,d_h)
        att_out = []
        for step in range(self.att_steps):
            ques = self.layer_que(context)  # ques(b,d_h)
            img = self.layer_img(img_emb)   # img(b,m,d_h)

            ques = tf.expand_dims(ques, axis=-2)  # ques(b,1,d_h)

            IQ = tf.nn.tanh(img + ques)  # IQ(b,m,d_h)
            # IQ = self.dropout(IQ)
            IQ_temp = self.layer_temp(IQ)  # IQ_temp(b,m,1)

            IQ_temp = tf.reshape(IQ_temp, [-1, IQ_temp.shape[1]])  # IQ_temp(b,m)

            p = tf.nn.softmax(IQ_temp)  # p(b,m)
            att_out.append(p)

            p_exp = tf.expand_dims(p, axis=-1)  # p_exp(b,m,1)

            att = tf.reduce_sum(p_exp * img_emb, axis=1)  # att(b,d_h)   p_exp(b,m,1)*img_emb(b,m,d_h)

            context += att

        return context, hidden_text, cell, att_out

class Decoder(Model):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super(Decoder, self).__init__()

        self.hid_dim = hid_dim
        self.vocab_size = vocab_size

        self.embedding = Embedding(vocab_size + 1, emb_dim)

        self.lstm = LSTM(hid_dim, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        self.fc = Dense(vocab_size + 1)

    def call(self, x, context, hidden_state, cell_state):
        embed = self.embedding(x)

        _, hidden_text, cell = self.lstm(embed, initial_state=[hidden_state, cell_state])


        # embed = tf.reshape(embed, (embed.shape[0], -1))
        # fusing = tf.concat((embed, hidden_text, context), -1)
        fusing = tf.concat((hidden_text, context), -1)

        prediction = self.fc(fusing)

        return prediction, hidden_text, cell

class VisLSTM(Model):
    def __init__(self, emb_dim=512, hidden_dim=512, vocab_size_que=3116, vocab_size_ans=2104, att_steps=2):
        super(VisLSTM, self).__init__()

        self.encoder = Encoder(vocab_size_que, emb_dim, hidden_dim, att_steps=att_steps)
        self.decoder = Decoder(vocab_size_ans, emb_dim, hidden_dim)

    def call(self, src, trg, image_info, boa_index=1, eoa_index=2):

        outputs = []
        context, hidden, cell, att_out = self.encoder(src, image_info)

        if trg is not None:  # train mode
            trg_len = trg.shape[1]
            inp = tf.expand_dims(trg[:, 0], 1)
            for t in range(1, trg_len):
                prediction, hidden, cell = self.decoder(inp, context, hidden, cell)
                outputs.append(prediction)

                inp = tf.expand_dims(trg[:, t], 1)
            return tf.stack(outputs, 1), att_out

        elif trg is None:  # test mode
            inp = tf.reshape([boa_index], (1, 1))
            prediction = boa_index
            while prediction != eoa_index and len(outputs) < 20:
                prediction, hidden, cell = self.decoder(inp, context, hidden, cell)
                prediction = tf.argmax(prediction, axis=-1)
                outputs.append(prediction)
                inp = tf.reshape(prediction, (1, 1))

            return tf.stack(outputs, 1), att_out


if __name__ == '__main__':

    BATCH_SIZE = 1
    EMB_SIZE = 512
    HIDDEN_SIZE = 512
    ATT_STEPS = 2
    QUE_VOCAB_SIZE = 3116
    ANS_VOCAB_SIZE = 2104

    # # test encoder
    # encoder = Encoder(3116, EMB_SIZE, HIDDEN_SIZE)
    x_text = tf.zeros((BATCH_SIZE, 20))
    #x_img = tf.zeros((BATCH_SIZE, 4096))
    x_img = tf.zeros((BATCH_SIZE,  14*14, HIDDEN_SIZE))

    # context, hidden, cell = encoder(x_text, x_img, 2)
    # print('*' * 30)
    # print('test encoder:')
    # print(context.shape, hidden.shape, cell.shape)
    # print('*'*30)
    #
    # # test decoder
    # decoder = Decoder(ANS_VOCAB_SIZE, EMB_SIZE, HIDDEN_SIZE)
    # d_x = tf.zeros((BATCH_SIZE, 1))
    # d_context = tf.zeros((BATCH_SIZE, 1024))
    # d_hidden = tf.zeros((BATCH_SIZE, 512))
    # d_cell = tf.zeros((BATCH_SIZE, 512))
    #
    # d_predict, d_hidden, d_cell = decoder(d_x, d_context, d_hidden, d_cell)
    # print('*' * 30)
    # print('test decoder')
    # print(d_predict.shape, d_hidden.shape, d_cell.shape)
    # print('*'*30)

    # test VisLSTM
    model = VisLSTM(emb_dim=EMB_SIZE, hidden_dim=HIDDEN_SIZE,
                    vocab_size_que=QUE_VOCAB_SIZE, vocab_size_ans=ANS_VOCAB_SIZE, att_steps=ATT_STEPS)
    src = tf.zeros((BATCH_SIZE, 20))   # question max length = 20
    trg = tf.zeros((BATCH_SIZE, 15))   # answer max length = 15
    out, _ = model(src, trg=None, image_info=x_img)
    print('*'*30)
    print('test VisLSTM')
    print(out.shape)
    print(len(_), _[0].shape)
    print('*'*30)
