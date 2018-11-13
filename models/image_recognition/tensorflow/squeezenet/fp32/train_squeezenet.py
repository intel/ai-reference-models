import os, time, json
import tensorflow as tf
import numpy as np
from deployment import model_deploy, arg_parsing, metrics, inputs
from tensorflow.python.client import timeline
import  networks
 
        
def _run(args):
    network = networks.catalogue[args.network](args)

    if args.cpu:
        deploy_config = _configure_deployment(
            num_clones = 1,
            clone_on_cpu = True)
        sess = tf.Session(config=tf.ConfigProto(               
                inter_op_parallelism_threads=args.num_inter_threads,
                intra_op_parallelism_threads=args.num_intra_threads))
            
    else:
        deploy_config = _configure_deployment(
            num_clones = args.num_gpus,
            clone_on_cpu = False)    
        sess = tf.Session(config=_configure_session(
                inter=args.num_inter_threads,
                intra=args.num_intra_threads))

    with tf.device(deploy_config.variables_device()):
        global_step = tf.train.create_global_step()

    with tf.device(deploy_config.optimizer_device()):
        lr = tf.train.polynomial_decay(
            args.learning_rate,
            global_step=global_step,
            end_learning_rate=args.min_lr,
            decay_steps= args.decay_steps,
            power=1,
            cycle=True)
        if args.optimizer.startswith('a'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        elif args.optimizer.startswith('r'):
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=lr,
                momentum=0.9,
                epsilon=1.0,)
        else:
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=lr,
                momentum=0.9,
                use_nesterov=True,)

    '''Inputs'''
    with tf.device(deploy_config.inputs_device()), tf.name_scope('inputs'):
        pipeline = inputs.Pipeline(args, sess)
        examples, labels = pipeline.data
        images = examples['image']

        image_splits = tf.split(
            value=images,
            num_or_size_splits=deploy_config.num_clones,
            name='split_images'
        )
        label_splits = tf.split(
            value=labels,
            num_or_size_splits=deploy_config.num_clones,
            name='split_labels'
        )

    '''Model Creation'''
    model_dp = model_deploy.deploy(
        config=deploy_config,
        model_fn=_clone_fn,
        optimizer=optimizer,
        kwargs={
            'images': image_splits,
            'labels': label_splits,
            'index_iter': iter(range(deploy_config.num_clones)),
            'network': network,
            'is_training': False  # pipeline.is_training
        }
    )

    '''Metrics'''
    with tf.name_scope('outputs'):
        train_metrics = metrics.Metrics(
            labels=labels,
            clone_logits=[clone.outputs['logits']
                            for clone in model_dp.clones],
            clone_predictions=[clone.outputs['predictions']
                            for clone in model_dp.clones],
            device=deploy_config.variables_device(),
            name='training'
        )
        validation_metrics = metrics.Metrics(
            labels=labels,
            clone_logits=[clone.outputs['logits']
                            for clone in model_dp.clones],
            clone_predictions=[clone.outputs['predictions']
                            for clone in model_dp.clones],
            device=deploy_config.variables_device(),
            name='validation',
            padded_data=True
        )
        validation_init_op = tf.group(
            pipeline.validation_iterator.initializer,
            validation_metrics.reset_op
        )
        train_op = tf.group(
            model_dp.train_op,
            train_metrics.update_op
        )

    '''Summaries'''
    with tf.device(deploy_config.variables_device()):
        train_writer = tf.summary.FileWriter(args.model_dir, sess.graph)
        eval_dir = os.path.join(args.model_dir, 'eval')
        eval_writer = tf.summary.FileWriter(eval_dir, sess.graph)
        tf.summary.scalar('accuracy', train_metrics.accuracy)
        tf.summary.scalar('loss', model_dp.total_loss)
        tf.summary.scalar('learning_rate',lr)
        all_summaries = tf.summary.merge_all()
  
    '''Model Checkpoints'''
    saver = tf.train.Saver(max_to_keep=args.keep_last_n_checkpoints)
    save_path = os.path.join(args.model_dir, 'model.ckpt')

    '''Model Initialization'''
    last_checkpoint = tf.train.latest_checkpoint(args.model_dir)
    if last_checkpoint:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver.restore(sess, last_checkpoint)
    else:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
    starting_step = sess.run(global_step)

    '''Save pb graph that C++ can load and run.'''
    tf.train.write_graph(sess.graph_def,'./graph/', 'sqz.pb', False)

    def _eval(args):
        dtime = []
        for i in range(args.eval_steps):
            try:
                if args.timeline >0 and i >0 and i % args.timeline == 0:
                    options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    t0 = time.time()                    
                    sess.run(fetches=validation_metrics.update_op,
                             feed_dict=pipeline.validation_data,
                             options=options,
                             run_metadata=run_metadata,)
                    duration = time.time() - t0
                    
                    fetched_timeline = timeline.Timeline(
                        run_metadata.step_stats)
                    chrome_trace = fetched_timeline.\
                        generate_chrome_trace_format()
                    with open('timeline/timeline_cpu_clone1_{0:03}.json'
                              .format(i), 'w') as f:
                        f.write(chrome_trace)
                else:
                    t0 = time.time()                    
                    sess.run(fetches=validation_metrics.update_op,
                             feed_dict=pipeline.validation_data,)
                    duration = time.time() - t0
                  
                if args.verbose: print('step {:03}: {:.4f} sec'.
                                       format(i, duration))
                dtime.append(duration)
           
            except:
                i = i - 1
                break

        num_examples = (i)*args.batch_size
        if len(dtime) <= 1:
            print('Only one warm-up step was executed! Run more steps')
        else:
            warmup = dtime[0]
            dtime  = dtime[1:]
            t_sum  = np.sum(dtime)
            t_mean = np.mean(dtime) 
            t_median=np.median(dtime)
            t_min  = np.min(dtime)
            t_max  = np.max(dtime)
            t_std  = np.std(dtime)

            """
            print('''{:.0f} batches x {:.0f} bs = total {:.0f} images
                throughput[avg] = {:.1f} ips 
                throughput[med] = {:.1f} ips
                latency[median] = {:>.4} ms
                latency[averge] = {:>.4} ms''' 
                    .format( i, args.batch_size, num_examples, 
                        num_examples/t_sum,
                        args.batch_size/t_median,
                        t_median*1000/args.batch_size,
                        t_sum*1000/num_examples))
            """
            print('''SqueezeNet Inference Summary:
            {:.0f} batches x {:.0f} bs = total {:.0f} images evaluated
            batch size = {}
            throughput[med] = {:.1f} image/sec
            latency[median] = {:>.4} ms
            '''.format( i, args.batch_size, num_examples, 
                        args.batch_size,
                        args.batch_size/t_median,
                        t_median*1000/args.batch_size))     
            
        # Reinitialize dataset and metrics after going through all validation
        # examples
        sess.run(validation_init_op)  

    def _train(args):
        '''Main Loop'''
        for train_step in range(starting_step, args.max_train_steps):
            sess.run(train_op, feed_dict=pipeline.training_data)

            '''Summary Hook'''
            if train_step % args.summary_interval == 0:
                results = sess.run(
                    fetches={'accuracy': train_metrics.accuracy,
                            'summary': all_summaries},
                    feed_dict=pipeline.training_data
                )
                train_writer.add_summary(results['summary'], train_step)

                print('*** Step {:<5}'.format(train_step))
                print('Train: acc= {:>.4}%'.format(results['accuracy']*100))

            '''Checkpoint Hooks'''
            if train_step % args.checkpoint_interval == 0:
                saver.save(sess, save_path, global_step)

            sess.run(train_metrics.reset_op)

            '''Eval Hook'''
            if train_step % args.validation_interval == 0:
                while True:
                    try:
                        sess.run(
                            fetches=validation_metrics.update_op,
                            feed_dict=pipeline.validation_data
                        )
                    except tf.errors.OutOfRangeError:
                        break
                results = sess.run({'accuracy': validation_metrics.accuracy})
                print('Validation: acc= {:>.4}%'.
                      format(results['accuracy']*100))

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag='accuracy',
                                     simple_value=results['accuracy']),
                ])
                eval_writer.add_summary(summary, train_step)

                # Reinitialize dataset and metrics
                sess.run(validation_init_op)
    
    if args.inference_only:
        _eval(args)
    else:
        _train(args)
 

def _clone_fn(images, labels, index_iter, network, is_training):
    clone_index = next(index_iter)
    images = images[clone_index]
    labels = labels[clone_index]

    unscaled_logits = network.build(images, is_training)
    tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                           logits=unscaled_logits)
    predictions = tf.argmax(unscaled_logits, 1, name='predictions')
    return {'logits': unscaled_logits, 'predictions': predictions,
            'images': images, }


def _configure_deployment(num_clones, clone_on_cpu):
    return model_deploy.DeploymentConfig(
        num_clones=num_clones, clone_on_cpu=clone_on_cpu,)


def _configure_session(inter,intra):
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=.85)
    return tf.ConfigProto(allow_soft_placement=True,
                          gpu_options=gpu_config,
                          inter_op_parallelism_threads=inter,
                          intra_op_parallelism_threads=intra)


def run(args=None):
    args = arg_parsing.ArgParser().parse_args(args)
    
    for xdir in ['log','timeline']:
        if not os.path.exists(xdir):
            os.makedirs(xdir)

    with tf.Graph().as_default():
        _run(args)

if __name__ == '__main__':
    run()
