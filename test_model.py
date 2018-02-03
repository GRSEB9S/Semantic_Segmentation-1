import tensorflow as tf
import helper

def test_model():
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    with tf.Session() as sess:
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'

        logits_operation_name = "final_upsampled_8x/BiasAdd"

        tf.saved_model.loader.load(sess, ["vgg16_semantic"], "./saved_model_epoch_30")

        graph = tf.get_default_graph()

        for op in tf.get_default_graph().get_operations():
            print (op.name) 

        vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
        vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

        print ("Logits ************************")
        logits_tensor = graph.get_operation_by_name(logits_operation_name).outputs[0]
        print (logits_tensor)
        print ("Inference")
        helper.save_inference_samples(runs_dir=runs_dir, data_dir=data_dir, sess=sess,image_shape=image_shape,
                                      logits=logits_tensor, keep_prob=vgg_keep_prob_tensor,
                                      input_image=vgg_input_tensor)

test_model()