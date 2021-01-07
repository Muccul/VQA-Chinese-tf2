from data_loader import load_traindata
from vis_lstm_san import VisLSTM
import tensorflow as tf

EMB_SIZE = 512
HIDDEN_SIZE = 512
ATT_STEPS = 2

IMG_SHAPE = (224, 224, 3)
VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE)
vgg16_extractor = tf.keras.Sequential(VGG16_MODEL.layers[:-5])
vgg16_extractor.trainable = False

dataset_train_ori, tokenizer_que, tokenizer_ans, max_que_train, max_ans_train = load_traindata()
vocab_size_que = len(tokenizer_que.word_counts)
vocab_size_ans = len(tokenizer_ans.word_counts)


model = VisLSTM(emb_dim=EMB_SIZE, hidden_dim=HIDDEN_SIZE,
                vocab_size_que=vocab_size_que, vocab_size_ans=vocab_size_ans,
                att_steps=ATT_STEPS)
checkpoint_dir = './weights'
checkpoint = tf.train.Checkpoint(model=model)
latest_file = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint.restore(latest_file)
model.trainable = False

img_path = ''
question_text = ''
question_seq = []
answer = ''
att = None