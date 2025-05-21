# codes adapted from https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git
# all copyright to https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git
import os
import time
import ctypes
import subprocess
import threading
import multiprocessing as mp
import asyncio

import numpy as np

from prover.utils import AttrDict


class TaskQueue(object):
    def __init__(self, batch_size=512, name='test'):
        self.name = name
        self.batch_size = batch_size
        self.manager = mp.Manager()
        self.waiting_list = self.manager.list()
        self.all_tasks_done = mp.Event()
        self.lock = mp.Lock()

        self._monitor_log = self.manager.list()
        self._monitor_thread = threading.Thread(target=self._monitor)
        self._monitor_thread.start()
    
    def _monitor(self):
        last_log_time = time.time()
        while not self.all_tasks_done.is_set():
            PERIOD = 20.0
            if time.time() - last_log_time >= PERIOD:
                with self.lock:
                    if len(self._monitor_log) > 0:
                        print('TaskQueue-{}: {} requests popped with avg bs {:.1f} in last {}s. {} waiting in queue.'.format(
                            self.name, np.sum(self._monitor_log), np.mean(self._monitor_log), PERIOD, len(self.waiting_list),
                        ))
                        self._monitor_log[:] = []
                last_log_time = time.time()
            time.sleep(1.0)
    
    def __len__(self):
        return len(self.waiting_list)
    
    def put(self, item):
        with self.lock:
            self.waiting_list.append(item)
    
    def get(self, no_wait=False):
        while not self.all_tasks_done.is_set():
            with self.lock:
                if len(self.waiting_list) > 0:
                    tasks = self.waiting_list[:self.batch_size]
                    self.waiting_list[:self.batch_size] = []
                    self._monitor_log.append(len(tasks))
                    return tasks
            if no_wait:
                break
            time.sleep(0.1)
        return None
    
    def close(self):
        self.all_tasks_done.set()
        self._monitor_thread.join()


class ProcessScheduler(object):
    def __init__(self, batch_size=512, name='test'):
        self.name = name
        self.manager = mp.Manager()
        self.batch_size = batch_size
        self.task_queue = TaskQueue(batch_size=batch_size, name=name)
        self.request_statuses = self.manager.dict()
        self.request_counter = mp.Value(ctypes.c_int32, 0)
        self.lock = mp.Lock()

    def submit_request(self, data):
        with self.lock:
            self.request_counter.value += 1
            request_id = self.request_counter.value
            self.request_statuses[request_id] = None
            self.task_queue.put((time.time(), request_id, data))
        return request_id
    
    def submit_all_request(self, data_list):
        request_id_list = [self.submit_request(data) for data in data_list]
        return request_id_list

    def get_request_status(self, request_id):
        with self.lock:
            response = self.request_statuses.get(request_id, None)
            if response is not None:
                self.request_statuses.pop(request_id)
            return response
    
    def get_request_outputs(self, request_id):
        """Synchronous version of get_request_outputs, blocks until result is returned"""
        while True:
            outputs = self.get_request_status(request_id)
            if outputs is not None:
                return outputs
            time.sleep(0.1)  # use time.sleep for blocking wait
    
    def get_all_request_outputs(self, request_id_list):
        """Synchronous version of get_all_request_outputs, blocks until all results are returned"""
        outputs_list = []
        for request_id in request_id_list:
            outputs_list.append(self.get_request_outputs(request_id))
        return outputs_list
    
    async def async_get_request_outputs(self, request_id):
        """Asynchronous version of get_request_outputs, does not block the event loop"""
        while True:
            outputs = self.get_request_status(request_id)
            if outputs is not None:
                return outputs
            await asyncio.sleep(0.1)  # use asyncio.sleep instead of time.sleep
    
    async def async_get_all_request_outputs(self, request_id_list):
        """Asynchronous parallel get all request outputs"""
        # parallel wait for all request results
        tasks = [self.async_get_request_outputs(request_id) for request_id in request_id_list]
        outputs_list = await asyncio.gather(*tasks)
        return outputs_list
    
    def close(self):
        self.task_queue.close()

# this is so terrible...
class Scheduler(object):
    def __init__(self, scheduler_dict):
        self._scheduler_dict = scheduler_dict
        for name, scheduler in scheduler_dict.items():
            self.__setattr__(name, scheduler)
            for key in dir(scheduler):
                if not key.startswith('_'):
                    self.__setattr__(f'{name}_{key}', scheduler.__getattribute__(key))
    
    def close(self):
        for _, scheduler in self._scheduler_dict.items():
            scheduler.close()
