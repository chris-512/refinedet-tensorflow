import os 
import scipy.misc 
import numpy as np 

import tensorflow as tf 
from model import RefineDet 

'''
https://github.com/tensorflow/models/tree/master/research/inception/inception
'''

def main(_):

    tf.logging.set_verbosity(tf.logging.DEBUG)

    if tf.test.is_built_with_cuda():
        pass
    
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto() 
    run_config.gpu_options.allow_growth = True



def train(dataset): 
    
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        
        # Calculate the learning rate schedule
        num_batches_per_epoch = (dataset.num_examples_per_epoch() / FLAGS.batch_size)

        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps. 
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)
        # Create an optimizer that performs gradient descent.
        opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY, 
                                        momentum=RMSPROP_MOMENTUM,
                                        epsilon=RMSPROP_EPSILON)
                                    
        # Get images and labels for ImageNet and split the batch across GPUs.
        assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')
        split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

        # Override the number of preprocessing threads to account for the increased 
        # number of GPU towers.
        num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus 
        images, labels = image_preprocessing.distorted_inputs(
            datasets,
            num_preprocess_threads=num_preprocess_threads)

        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes() + 1

        # Split the batch of images and labels for towers. 
        images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
        labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)

        # Calculate the gradients for each model tower.
        tower_grads = []
        reuse_variables = None
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
                    # Force all variables to reside on the CPU.
                    with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                        # Calculate the loss for one tower of the ImageNet model. This
                        # function constructs the entire ImageNet model but shares the
                        # variables across all towers.
                        loss = _tower_loss(images_splits[i], labels_splits[i], num_classes,
                                            scope, reuse_variables)

                    # Reuse variables for the next tower.
                    reuse_variables = True

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope) 

                    # Retain the Batch Normalization updates operations only from the 
                    # final tower. Ideally, we should grab the updates from all towers
                    # but these stats accumulate extremely fast so we can ignore the 
                    # other stats from the other towers without significant detriment.
                    batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                                        scope) 

                    # Calculate the gradients for the batch of data on this ImageNet tower.
                    grads = opt.compute_gradients(loss) 

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads) 
            
            # We must calculate the mean of each gradient. Note that this is the 
            # synchronization point across all towers. 
            grads = _average_gradients(tower_grads) 

            # Add a summaries for the input processing and global_step.
            summaries.extend(input_summaries)

            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', lr))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(
                        tf.summary.histogram(var.op.name + '/gradients', grad))
                
            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables. 
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var)) 
            
            # Track the moving averages of all trainable variables.
            # Note that we maintain a "double-average" of the BatchNormalization
            # global statistics. This is more complicated then need be but we employ
            # this for backward compatibility with our previous models. 
            variable_averages = tf.train.ExponentialMovingAverage(
                inception.MOVING_AVERAGE_DECAY, global_step)
            
            # Another possibility is to use tf.slim.get_variables().
            variables_to_average = (tf.trainable_variables() + 
                                    tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average) 

            # Group all updates to into a single train op.
            batchnorm_updates_op = tf.group(*batchnorm_updates) 
            train_op = tf.group(apply_gradient_op, variables_averages_op, 
                                batchnorm_updates_op) 
            
            # Create a saver.
            saver = tf.train.Saver(tf.global_variables())

            # Build the summary operation from the last tower summaries.
            summary_op = tf.summary.merge(summaries)

            # Build an initialization operation to run below. 
            init = tf.global_variables_initializer() 

            # Start running operations on the Graph. allow_soft_placement must be set to 
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, 
                log_device_placement=FLAGS.log_device_placement
            ))
            sess.run(init) 

            if FLAGS.pretrained_model_checkpoint_path:
                assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path) 
                variables_to_restore = tf.get_collection(
                    slim.variables.VARIABLES_TO_RESTORE)
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path) 
                print('%s: Pre-trained model restored from %s' % 
                (datetime.now(), FLAGS.pretrained_model_checkpoint_path)) 

            # Start the queue runners. 
            tf.train.start_queue_runners(sess=sess) 

            summary_writer = tf.summary.FileWriter(
                FLAGS.train_dir,
                graph=sess.graph
            )

            for step in range(FLAGS.max_steps):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time 

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    examples_per_sec = FLAGS.batch_size / float(duration) 
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value, 
                                        examples_per_sec, duration))

                if step % 100 == 0:
                    summary_str = sess.run(summary_op) 
                    summary_writer.add_summary(summary_str, step) 

                # Save the model checkpoint periodically.
                if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_ste=step)


    #net = build_refinedet('train', cfg['min_dim'], cfg['num_classes'])

    



if __name__ == '__main__':
    tf.app.run()