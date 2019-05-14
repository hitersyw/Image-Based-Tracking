import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import datetime

from .utils import utils, helpers
from .builders import model_builder

class Predictor:
    """!
    Class used to query the trained neural network. Currently only segmentation of surgical instruments is supported.
    This class relies on the <a href="https://github.com/GeorgeSeif/Semantic-Segmentation-Suite">Semantic Segmentation Suite by George Seif</a>.

    If you want to feed in your own trained net, see his GitHub page for information on how to train the net.
    In this case make sure to provide the label info file 'class_dict.csv' in '/src/Segmentation/utils/'.
    """

    def __init__(self, model, checkpoint_path):
        """!
        Initializes the neural net by loading the pretrained model.

        @param model The name of the model to use. Make sure your pretrained model was trained using the same model.
        @param checkpoint_path The path to the pretrained model file.
        """
        label_info_path = "./src/Segmentation/utils/class_dict.csv"
        class_names_list, self.label_values = helpers.get_label_info(label_info_path)

        num_classes = len(self.label_values)
        # Initializing network
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess=tf.Session(config=config)

        self.net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
        self.net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

        self.network, _ = model_builder.build_model(model, net_input=self.net_input,
                                                num_classes=num_classes,
                                                crop_width=0,
                                                crop_height=0,
                                                is_training=False)
        self.sess.run(tf.global_variables_initializer())

        saver=tf.train.Saver(max_to_keep=1000)
        tf.reset_default_graph()
        saver.restore(self.sess, checkpoint_path)


    def predict(self, image):
        """!
        Queries the network loaded in init using the given image as input.

        @param image The image to query

        @return An image with the predicted labels according to the given image.
        """
        height, width, _ = image.shape

        loaded_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(loaded_image, (width, height))
        input_image = np.expand_dims(np.float32(resized_image[:height, :width]),axis=0)/255.0

        time1 = datetime.datetime.now()

        output_image = self.sess.run(self.network,feed_dict={self.net_input:input_image})

        time2 = datetime.datetime.now()
        print("time for query: ", time2 - time1)

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)

        out_vis_image = helpers.colour_code_segmentation(output_image, self.label_values)
        return cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
