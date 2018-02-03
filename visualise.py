import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from tensorflow.core.protobuf import saved_model_pb2

def graph_visualize():


    # Path to vgg model
    #data_dir = './data'
    #vgg_path = os.path.join(data_dir, 'vgg')
    
    data_dir = './'
    vgg_path = os.path.join(data_dir, 'saved_model_epoch_30')

    with tf.Session() as sess:
        model_filename = os.path.join(vgg_path, 'saved_model.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
    LOGDIR = 'logs_tensorboard/.'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    print ("Printed")


graph_visualize()