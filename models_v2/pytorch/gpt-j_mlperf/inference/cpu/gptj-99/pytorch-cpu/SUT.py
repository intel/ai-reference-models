#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from argparse import ArgumentParser
import os
import sys
import time
import multiprocessing as mp
import array
import numpy as np
import threading
import subprocess
import logging
import collections

from item import InputItem, OutputItem
import thread_binder

#from dataset import Dataset
#from backend import Backend

import mlperf_loadgen as lg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SUT")

class Consumer(mp.Process):
    def __init__(self, model_checkpoint_path="", precision="int8", quantized_model="", dataset_path="", input_queue=None, out_queue=None, lock=None, cond_var=None, init_counter=None, proc_idx=None, start_core_idx=0, cpus_per_proc=56, workers_per_proc=1, warmup=False, total_sample_count=1000, pad_inputs=False, input_lens=None, batch_size=1, bind_logical_cores=True, logical_cores_start=0):

        mp.Process.__init__(self)
        self.num_workers = workers_per_proc
        self.task_queue = input_queue
        self.out_queue = out_queue
        self.lock = lock
        self.init_counter = init_counter
        self.proc_idx = proc_idx
        self.num_cores = cpus_per_proc
        self.start_core_idx = start_core_idx
        self.end_core_idx = start_core_idx + self.num_cores - 1
        self.affinity = list(range(self.start_core_idx, self.start_core_idx + self.num_cores))
        if bind_logical_cores:
            self.affinity.extend(list(range(self.start_core_idx + logical_cores_start, self.start_core_idx + logical_cores_start + self.num_cores)))

        self.dataset_path = dataset_path

        self.cpus_per_worker = self.num_cores // self.num_workers
        self.workers = []
        self.out_queue = out_queue
        self.warmup = warmup
        self.latencies = collections.defaultdict(list)

        self.total_sample_count = total_sample_count
        self.pad_inputs = pad_inputs

        self.model_checkpoint_path = model_checkpoint_path
        self.precision = precision
        self.quantized_model = quantized_model
        self.cond_var = cond_var

        self.input_lens = input_lens
        self.batch_size = [int(batch) for batch in batch_size.split(",")] if isinstance(batch_size, str) else [batch_size] * self.num_workers

        self.bind_logical_cores = bind_logical_cores
        self.logical_cores_start = logical_cores_start
        

    def doWarmup(self, worker_name, batch_size=1):
        warmup_data = self.data_obj.getWarmupSamples()
        log.info("Starting warmup")
        input_ids, input_len, attention_mask = warmup_data[0]

        input_ids = input_ids.repeat(batch_size, 1)
        attention_mask = attention_mask.repeat(batch_size, 1)

        output = self.model.predict(input_ids, attention_mask)
        output = self.model.predict(input_ids, attention_mask)
        for i, (input_ids, input_len, attention_mask) in enumerate(warmup_data):
            input_ids = input_ids.repeat(batch_size, 1)
            attention_mask = attention_mask.repeat(batch_size, 1)

            output = self.model.predict(input_ids, attention_mask)

        log.info("Worker {} Warmup Completed".format(worker_name))
        with self.cond_var:
            self.init_counter.value += 1
            self.cond_var.notify()


    def handleTasks(self, i, task_queue, result_queue, pid, start_core, num_cores):
        worker_name = str(pid) + "-" + str(i)
        cores_affinity = list(range(start_core, start_core + num_cores))
        if self.bind_logical_cores:
            logical_cores_affinity = list(range(start_core + self.logical_cores_start, start_core + self.logical_cores_start + num_cores))
            cores_affinity.extend(logical_cores_affinity)

        thread_binder.set_worker_affinity(cores_affinity)
        log.info("Bound worker {} to cores {}".format(worker_name, cores_affinity))

        idx = min(i, len(self.batch_size) - 1)
        worker_batchsize = self.batch_size[idx]
        # Do Warmup
        if self.warmup:
            self.doWarmup(worker_name, worker_batchsize)

        else:
            with self.cond_var:
                self.init_counter.value += 1
                self.cond_var.notify_all()

        query_id_list = []
        sample_index_list = []

        while True:
            try:
                next_task = task_queue.get()
                if next_task is None:
                    if len(query_id_list) > 0:
                        input_ids, input_seq_lens, attention_mask = self.data_obj.getSamples(sample_index_list)

                        output = self.model.predict(input_ids, attention_mask=attention_mask)
                        result = self.data_obj.postProcess(query_id_list, sample_index_list, output, input_seq_lens)
                        result_queue.put(result)
                        task_queue.task_done()

                    log.info("Exiting worker thread : {}".format(worker_name))
                    break

                query_id = next_task.query_id_list
                sample_index = next_task.sample_index_list

                query_id_list.extend(query_id)
                sample_index_list.extend(sample_index)

                if len(query_id_list)==worker_batchsize:
                    input_ids, input_seq_lens, attention_mask = self.data_obj.getSamples(sample_index_list)

                    output = self.model.predict(input_ids, attention_mask=attention_mask)
                    result = self.data_obj.postProcess(query_id_list, sample_index_list, output, input_seq_lens)
                    #log.info("{} batch {} inference took {}".format(worker_name, len(result.result), round(time.time() - t0,3)))

                    result_queue.put(result)
                    task_queue.task_done()

                    query_id_list = []
                    sample_index_list = []

            except Exception as ex:
                # Error occured
                log.error(ex)
                break
                self.terminate()
                sys.exit(1)


    def run(self):
        os.sched_setaffinity(0, self.affinity)

        from backend import Backend
        self.model = Backend(model_checkpoint=self.model_checkpoint_path,
                precision=self.precision,
                quantized_model=self.quantized_model
                )

        # Load model
        log.info("Loading model")
        self.model.loadModel()
        
        from dataset import Dataset
        self.data_obj = Dataset(self.dataset_path, model_checkpoint_path=self.model_checkpoint_path, total_sample_count=self.total_sample_count, pad_inputs=self.pad_inputs)
        
        # Load Dataset
        log.info("Loading Dataset")
        self.data_obj.loadDataset()

        # Get input sequence lengths
        if self.proc_idx==0:
            with self.cond_var:
                for input_len in self.data_obj.getInputLengths():
                    self.input_lens.append(input_len)
                self.cond_var.notify()

        start_core = self.start_core_idx
        cores_left = self.num_cores
        cores_rem = self.num_cores - self.num_workers * self.cpus_per_worker
        for i in range(self.num_workers):
            log.info("Creating worker {}-{}".format(os.getpid(),i))
            worker_cores = self.cpus_per_worker + max(0, min(1, cores_rem)) #min(self.cpus_per_worker, cores_left)
            cores_left -= self.cpus_per_worker
            
            worker = mp.Process(target=self.handleTasks, args=(i, self.task_queue, self.out_queue, self.pid, start_core, worker_cores))

            self.workers.append(worker)
            start_core += worker_cores #self.cpus_per_worker
            cores_rem -= 1

        for w in self.workers:
            w.start()

        for w in self.workers:
            w.join()

        log.info("{} : Exiting consumer process".format(os.getpid()))


class SUT(object):
    def __init__(self, num_proc, cpus_per_proc, model_checkpoint_path, initial_core=0, batch_size=1, dataset_path=None, workers_per_proc=1, warmup=False, precision="int8", quantized_model=None, total_sample_count=1000, pad_inputs=False, bind_logical_cores=True, logical_cores_start=0, workers_proc_alloc=None, batch_proc_alloc=None):

        self.num_proc = num_proc
        self.cpus_per_proc = cpus_per_proc
        self.initial_core = initial_core
        #self.workers_per_proc = workers_per_proc
        self.warmup = warmup

        self.workers_proc_alloc = workers_proc_alloc
        self.batch_proc_alloc = batch_proc_alloc

        self.total_workers = self.num_proc * workers_per_proc
        if self.workers_proc_alloc is not None:
            self.total_workers = sum(self.workers_proc_alloc)
        else:
            self.workers_proc_alloc = [workers_per_proc] * self.num_proc

        self.workers_per_proc = self.workers_proc_alloc
        self.num_proc = len(self.workers_proc_alloc)
        self.procs = [None] * self.num_proc

        self.model_checkpoint_path = model_checkpoint_path
        self.precision = precision
        self.quantized_model = quantized_model

        self.batch_size = batch_size
        if self.batch_proc_alloc is None:
            self.batch_proc_alloc = [self.batch_size] * self.num_proc

        self.dataset_path = dataset_path

        self.total_sample_count = total_sample_count
        self.pad_inputs = pad_inputs

        self.lock = mp.Lock()
        self.init_counter = mp.Value("i", 0)
        self.input_queue = mp.JoinableQueue()
        self.output_queue = mp.Queue()
        
        self.cv = mp.Condition(lock=self.lock)

        from multiprocessing import Manager
        
        self.input_lens = Manager().list([])

        self.bind_logical_cores = bind_logical_cores
        self.logical_cores_start = logical_cores_start


    def flushQueries(self):
        pass

    def processLatencies(self, latencies):
        pass

    def loadSamplesToRam(self, query_samples):
        pass

    def unloadSamplesFromRam(self, query_samples):
        pass

    def stopSUT(self):
        """ Stops processes and threads and exit """

        for _ in range(self.total_workers):
            self.input_queue.put(None)

        for proc in self.procs:
            proc.join()

        self.output_queue.put(None)

    def startSUT(self):
        """ Creates and Starts the processes and threads"""

        # Create processes
        self.createProcesses()

        # Start processes
        log.info("Starting processes")
        for proc in self.procs:
            proc.start()
        
        # Wait for all consumers to be ready (including if they're warming up)
        with self.cv:
            self.cv.wait_for(lambda : self.init_counter.value==self.total_workers)

        # Start Loadgen response thread
        self.response_thread = threading.Thread(target=self.responseLoadgen)
        self.response_thread.start()

    def responseLoadgen(self):

        while True:
            next_task = self.output_queue.get()
            
            if next_task is None:
                log.info('Exiting response thread')
                break

            query_id_list = next_task.query_id_list
            processed_result = next_task.result
            array_type_code = next_task.array_type_code
            batch_size = len(query_id_list)
            
            for id, out in zip(query_id_list, processed_result):
                response_array = array.array(array_type_code, out.tobytes())
                bi = response_array.buffer_info()
                responses = [lg.QuerySampleResponse(id, bi[0], bi[1]*response_array.itemsize)]
                lg.QuerySamplesComplete(responses)

    def createProcesses(self):
        """ Create 'mp' instances or processes"""

        start_core = self.initial_core
        for proc_idx in range(self.num_proc):
            self.procs[proc_idx] = Consumer(self.model_checkpoint_path, self.precision, self.quantized_model, self.dataset_path, self.input_queue, self.output_queue, self.lock, self.cv, self.init_counter, proc_idx, start_core, self.cpus_per_proc, self.workers_proc_alloc[proc_idx], warmup=self.warmup, total_sample_count = self.total_sample_count, pad_inputs=self.pad_inputs, input_lens=self.input_lens if proc_idx==0 else None, batch_size = self.batch_proc_alloc[proc_idx], bind_logical_cores=self.bind_logical_cores, logical_cores_start=self.logical_cores_start)

            start_core += self.cpus_per_proc

    def issueQueries(self, query_samples):
        """ Receives queries and adds them to queue for processing"""
        # TODO: Implement Server logic in separate issueQuery

        num_samples = len(query_samples)
        ids = []        # Response Ids
        indexes = []    # Sample Indexes
        input_token_ids = []

        if num_samples > 1:
            query_samples.sort(key=lambda x : self.input_lens[x.index])

        for i in range( num_samples):
            item = InputItem([query_samples[i].id], [query_samples[i].index])
            self.input_queue.put(item)

        if num_samples > 1: # For Offline
            for _ in range(self.total_workers):
                self.input_queue.put(None)



