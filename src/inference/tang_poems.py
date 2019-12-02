import collections
import os
import sys
import numpy as np
import tensorflow as tf
from models.model import rnn_model
from dataset.poems import process_poems, generate_batch
import heapq
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

FLAGS_batch_size = 64
FLAGS_learning_rate = 0.01
FLAGS_checkpoints_dir = os.path.abspath('./checkpoints/poems/')
FLAGS_file_path = os.path.abspath('./dataset/data/poems.txt')
FLAGS_model_prefix = 'poems'
FLAGS_epochs = 50

start_token = 'G'
end_token = 'E'

def run_training():
    if not os.path.exists(os.path.dirname(FLAGS_checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS_checkpoints_dir))
    if not os.path.exists(FLAGS_checkpoints_dir):
        os.mkdir(FLAGS_checkpoints_dir)

    poems_vector, word_to_int, vocabularies = process_poems(FLAGS_file_path)
    batches_inputs, batches_outputs = generate_batch(FLAGS_batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS_batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS_batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS_learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS_checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, FLAGS_epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS_batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                    n += 1
                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                    
                if epoch % 6 == 0:
                    saver.save(sess, './model/', global_step=epoch)
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS_checkpoints_dir, FLAGS_model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def to_word(predict, vocabs):
    return vocabs[np.argmax(predict)]

def gen_poem(begin_word):
    batch_size = 1
    print('[INFO] loading corpus from %s' % FLAGS_file_path)
    poems_vector, word_int_map, vocabularies = process_poems(FLAGS_file_path)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=FLAGS_learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, './model/-48')

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        poem = ''
        i = 1
        while word != end_token and i < 50 and word != ' ':
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)
            i += 1
        return poem

def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


def main(is_train):
    if is_train:
        print('[INFO] train tang poem...')
        run_training()
    else:
        print('[INFO] write tang poem...')

        begin_word = input('输入起始字:')
        poem2 = gen_poem(begin_word)
        pretty_print_poem(poem2)

if __name__ == '__main__':
    tf.app.run()