import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import datetime

from .utils import utils, helpers
from .builders import model_builder

class Predictor:
    """!
    Class used to query the trained neural net. Currently only segmentation of surgical instruments is supported.
    This class relies on the <a href="https://github.com/GeorgeSeif/Semantic-Segmentation-Suite">Semantic Segmentation Suite by George Seif</a>.
    If you want to feed in your own trained net, see his GitHub page for information on how to train the net.
    """

    def predict(self, image, model, checkpoint_path):
        """!
        Queries the net with the given image.

        @param image The image to query
        @param model The model to use
        @param checkpoint_path The path to the pretrained network to use.

        @return An image with the predicted labels according to the given image.
        """
        height, width, _ = image.shape
        label_info_path = "./src/Segmentation/utils/class_dict.csv"
        class_names_list, label_values = helpers.get_label_info(label_info_path)

        num_classes = len(label_values)

        # Initializing network
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.Session(config=config)

        net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
        net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

        network, _ = model_builder.build_model(model, net_input=net_input,
                                                num_classes=num_classes,
                                                crop_width=width,
                                                crop_height=height,
                                                is_training=False)

        sess.run(tf.global_variables_initializer())

        saver=tf.train.Saver(max_to_keep=1000)
        tf.reset_default_graph()
        saver.restore(sess, checkpoint_path)

        loaded_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(loaded_image, (width, height))
        input_image = np.expand_dims(np.float32(resized_image[:height, :width]),axis=0)/255.0

        time1 = datetime.datetime.now()
        output_image = sess.run(network,feed_dict={net_input:input_image})
        time2 = datetime.datetime.now()
        print("time for query: ", time2 - time1)
        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)

        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
        return cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
