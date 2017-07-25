# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import thread



def render_with_model_file(
        image_file,                                         # image to render
        model_file,                                         # the trained model to gennerate output file
        loss_model='vgg_16',                               # vgg_16 ,vgg_19 or imagenet
        output_file_path='/temp/smg/transfer/'          # output dir for rendered image
):
    lock = thread.allocate_lock()
    lock.acquire()

    file_name = os.path.basename(image_file).split('.')[0]
    model_name = os.path.basename(model_file).split('.')[0]


    # Make sure 'generated' directory exists.
    generated_file = output_file_path + file_name + '_' + model_name
    if os.path.exists(output_file_path) is False:
        os.makedirs(output_file_path)

    print('image_file', image_file)
    print('model_file', model_file)
    print('gennerated_path,', generated_file)

    # Get image's height and width.
    height = 0
    width = 0
    with open(image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
                generated_file = generated_file + '.png'
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
                generated_file = generated_file + '.jpg'
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                loss_model,
                is_training=False)
            image = reader.get_image(image_file, height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            model_file = os.path.abspath(model_file)
            saver.restore(sess, model_file)



            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Save file to  %s' % generated_file)
    lock.release()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    render_with_model_file('9.jpg','vgg_16','models/mosaic.model')
