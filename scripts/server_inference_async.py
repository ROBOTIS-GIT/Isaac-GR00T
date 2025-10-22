#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

from dataclasses import dataclass
from io import BytesIO
from typing import Callable
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch
import zmq


class ZmqInferenceServer:
    def __init__(
            self,
            server_address: str,
            port: int = 5555):

        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f'tcp://{server_address}:{port}')
        self._callback_group = {}
        self.policy = None
        
        # Thread Pool for inference (single worker is sufficient)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference')
        
        # Async inference management
        self.inference_tasks = {}  # task_id -> {status, result, future, timestamp}
        self.inference_lock = threading.Lock()
        self.task_cleanup_threshold = 100  # Clean up after this many completed tasks
        self.task_age_threshold = 300  # Clean up tasks older than 5 minutes

        self.add_callback(name='ping', callback=self._ping_callback)
        self.add_callback(name='kill', callback=self._kill_server_callback)
        self.add_callback(name='load_policy', callback=self._load_policy_callback)
        self.add_callback(name='unload_policy', callback=self._unload_policy_callback)
        self.add_callback(name='start_inference', callback=self._start_inference_callback)
        self.add_callback(name='stop_inference', callback=self._stop_inference_callback)
        self.add_callback(name='check_inference', callback=self._check_inference_callback)
        self.add_callback(name='get_inference_result', callback=self._get_inference_result_callback)

    def add_callback(
            self,
            name: str,
            callback: Callable):
        self._callback_group[name] = callback
    
    def _cleanup_old_tasks(self):
        """Clean up old completed tasks to prevent memory leak"""
        current_time = time.time()
        tasks_to_delete = []
        
        # Find tasks to delete (must be called with lock held)
        for task_id, task in self.inference_tasks.items():
            if task['status'] in ['completed', 'error', 'stopped']:
                # Delete if older than threshold
                if current_time - task['timestamp'] > self.task_age_threshold:
                    tasks_to_delete.append(task_id)
        
        # Also check if we have too many completed tasks
        completed_tasks = [
            tid for tid, task in self.inference_tasks.items() 
            if task['status'] in ['completed', 'error', 'stopped']
        ]
        
        if len(completed_tasks) > self.task_cleanup_threshold:
            # Delete oldest completed tasks
            sorted_tasks = sorted(
                [(tid, self.inference_tasks[tid]['timestamp']) for tid in completed_tasks],
                key=lambda x: x[1]
            )
            # Keep only the newest 50
            tasks_to_delete.extend([tid for tid, _ in sorted_tasks[:-50]])
        
        # Delete tasks
        for task_id in set(tasks_to_delete):
            if task_id in self.inference_tasks:
                del self.inference_tasks[task_id]
        
        if tasks_to_delete:
            print(f'Cleaned up {len(tasks_to_delete)} old inference tasks')

    def convert_dict_to_bytes(self, data: dict) -> bytes:
        bytes_buffer = BytesIO()
        torch.save(data, bytes_buffer)
        return bytes_buffer.getvalue()

    def convert_dict_from_bytes(self, data: bytes) -> dict:
        bytes_buffer = BytesIO(data)
        dict_data = torch.load(bytes_buffer, weights_only=False)
        return dict_data

    def _ping_callback(self, data) -> dict:
        return {'status': 'ok', 'message': 'Server is running'}

    def _kill_server_callback(self, data):
        self.running = False

    def _load_policy_callback(self, data: dict) -> dict:
        if (
            'policy_type' not in data or
            'policy_path' not in data or
            'robot_type' not in data
        ):
            return {
                'status': 'error',
                'message': "Missing required fields: 'policy_type', 'policy_path', 'robot_type'"
            }

        try:
            # If policy already exists, unload it first
            if self.policy is not None:
                print("Unloading existing policy before loading new one...")
                import gc
                
                # Move model to CPU to free GPU memory
                if hasattr(self.policy, 'model'):
                    self.policy.model.cpu()
                
                # Delete policy
                del self.policy
                self.policy = None
                if 'get_action' in self._callback_group:
                    del self._callback_group['get_action']
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print("Previous policy unloaded successfully")
            
            # Force clear GPU memory before loading new policy
            if torch.cuda.is_available():
                print("Clearing GPU memory before loading new policy...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            if data['policy_type'] == 'GR00T_N1_5':
                from gr00t.experiment.data_config import load_data_config
                from gr00t.model.policy import Gr00tPolicy
                data_config = load_data_config(data['robot_type'])
                self.policy = Gr00tPolicy(
                    model_path=data['policy_path'],
                    modality_config=data_config.modality_config(),
                    modality_transform=data_config.transform(),
                    embodiment_tag='new_embodiment',
                    denoising_steps=data.get('denoising_steps', 4),
                )
                self.add_callback('get_action', self.policy.get_action)
                return {
                    'status': 'ok',
                    'message': 'Policy loaded successfully'
                }
            elif data['policy_type'] == 'GR00T_N1_5_TRT':
                from gr00t.experiment.data_config import load_data_config
                from gr00t.model.policy import Gr00tPolicy
                from deployment_scripts.trt_model_forward import setup_tensorrt_engines
                data_config = load_data_config(data['robot_type'])
                self.policy = Gr00tPolicy(
                    model_path=data['policy_path'],
                    modality_config=data_config.modality_config(),
                    modality_transform=data_config.transform(),
                    embodiment_tag='new_embodiment',
                    denoising_steps=data.get('denoising_steps', 4),
                )
                trt_path = data['policy_path'] + '_engine'
                setup_tensorrt_engines(self.policy, trt_path)

                self.add_callback('get_action', self.policy.get_action)
                return {
                    'status': 'ok',
                    'message': 'Policy loaded successfully'
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Policy not supported yet'
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Failed to load policy: {e}"
            }

    def _unload_policy_callback(self, data) -> dict:
        import gc
        
        if self.policy is None:
            return {'status': 'error', 'message': 'No policy loaded'}
        
        print("Unloading policy and freeing memory...")
        
        # Move model to CPU first to free GPU memory
        try:
            if hasattr(self.policy, 'model'):
                print("Moving model to CPU...")
                self.policy.model.cpu()
                
                # Clean up TensorRT engines if present
                if hasattr(self.policy.model, 'backbone'):
                    backbone = self.policy.model.backbone
                    for engine_name in ['vit_engine', 'llm_engine']:
                        if hasattr(backbone, engine_name):
                            print(f"Cleaning up TensorRT {engine_name}...")
                            delattr(backbone, engine_name)
                
                if hasattr(self.policy.model, 'action_head'):
                    action_head = self.policy.model.action_head
                    for engine_name in ['vlln_vl_self_attention_engine', 'action_encoder_engine', 
                                       'action_decoder_engine', 'DiT_engine', 'state_encoder_engine']:
                        if hasattr(action_head, engine_name):
                            print(f"Cleaning up TensorRT {engine_name}...")
                            delattr(action_head, engine_name)
        except Exception as e:
            print(f"Error during model cleanup: {e}")
            import traceback
            traceback.print_exc()
        
        # Delete policy reference
        del self.policy
        self.policy = None
        if 'get_action' in self._callback_group:
            del self._callback_group['get_action']
        
        # Clear all pending inference tasks
        with self.inference_lock:
            self.inference_tasks.clear()
        
        # Force multiple rounds of garbage collection
        print("Running garbage collection...")
        for i in range(5):
            collected = gc.collect()
            print(f"  GC round {i+1}: collected {collected} objects")
        
                # Clear CUDA cache if available
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force CUDA to release memory back to OS
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except:
                pass
            
            # Get memory info
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
        return {'status': 'ok', 'message': 'Policy unloaded and memory cleaned successfully'}

    def _start_inference_callback(self, data: dict) -> dict:
        """Start async inference and return task ID immediately"""
        if self.policy is None:
            return {
                'status': 'error',
                'message': 'No policy loaded'
            }
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task entry
        with self.inference_lock:
            self.inference_tasks[task_id] = {
                'status': 'processing',
                'result': None,
                'future': None,
                'timestamp': time.time()
            }
            
            # Clean up old completed tasks
            self._cleanup_old_tasks()
        
        # Start inference using thread pool
        def run_inference():
            try:
                start_time = time.time()
                result = self.policy.get_action(data)
                end_time = time.time()
                print(f'Inference task {task_id} completed in {end_time - start_time:.4f} seconds')
                with self.inference_lock:
                    if task_id in self.inference_tasks:
                        self.inference_tasks[task_id]['status'] = 'completed'
                        self.inference_tasks[task_id]['result'] = result
            except Exception as e:
                with self.inference_lock:
                    if task_id in self.inference_tasks:
                        self.inference_tasks[task_id]['status'] = 'error'
                        self.inference_tasks[task_id]['result'] = {'error': str(e)}
        
        # Submit to thread pool (reuses existing thread)
        future = self.executor.submit(run_inference)
        
        with self.inference_lock:
            self.inference_tasks[task_id]['future'] = future
        
        return {
            'status': 'ok',
            'task_id': task_id,
            'message': 'Inference started'
        }
    
    def _stop_inference_callback(self, data: dict) -> dict:
        """Stop all inference tasks and clean up thread pool (but keep policy loaded)"""
        import gc
        
        print("Stopping all inference tasks...")
        
        # Step 1: Mark all tasks as stopped
        with self.inference_lock:
            for task_id, task in self.inference_tasks.items():
                if task['status'] == 'processing':
                    task['status'] = 'stopped'
                    task['result'] = {'error': 'Inference was stopped by user'}
            
            stopped_count = len(self.inference_tasks)
            self.inference_tasks.clear()
        
        # Step 2: Shutdown thread pool and recreate it
        print("Shutting down thread pool...")
        self.executor.shutdown(wait=True, cancel_futures=True)
        print("Thread pool shut down. Creating new thread pool...")
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference')
        
        # Force garbage collection
        gc.collect()
        
        print(f"Stopped {stopped_count} inference task(s). Policy remains loaded.")
        
        return {
            'status': 'ok',
            'message': f'Stopped {stopped_count} inference task(s). Use unload_policy to free GPU memory.'
        }

    def _check_inference_callback(self, data: dict) -> dict:
        """Check if inference task is completed without heavy data transfer"""
        task_id = data.get('task_id')
        if not task_id:
            return {
                'status': 'error',
                'message': 'task_id required'
            }
        
        with self.inference_lock:
            if task_id not in self.inference_tasks:
                return {
                    'status': 'error',
                    'message': 'Task not found'
                }
            
            task_status = self.inference_tasks[task_id]['status']
            
        return {
            'status': 'ok',
            'task_status': task_status,
            'is_ready': task_status in ['completed', 'error']
        }

    def _get_inference_result_callback(self, data: dict) -> dict:
        """Get inference result and clean up task"""
        task_id = data.get('task_id')
        if not task_id:
            return {
                'status': 'error',
                'message': 'task_id required'
            }
        
        with self.inference_lock:
            if task_id not in self.inference_tasks:
                return {
                    'status': 'error',
                    'message': 'Task not found'
                }
            
            task = self.inference_tasks[task_id]
            if task['status'] == 'processing':
                return {
                    'status': 'error',
                    'message': 'Inference still processing'
                }
            
            result = task['result']
            # Clean up completed task
            del self.inference_tasks[task_id]
            
        if task['status'] == 'error':
            return {
                'status': 'error',
                'message': f"Inference failed: {result.get('error', 'Unknown error')}"
            }
        
        return result

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f'Server is ready and listening on {addr}')
        while self.running:
            try:
                message = self.socket.recv()
                request = self.convert_dict_from_bytes(message)
                command = request.get('command', 'get_action')

                if command not in self._callback_group:
                    error_response = {
                        'status': 'error',
                        'message': f'Unknown command: {command}', 
                    }
                    self.socket.send(self.convert_dict_to_bytes(error_response))
                    continue

                callback = self._callback_group[command]
                result = (
                    callback(request.get('data', {}))
                )
                self.socket.send(self.convert_dict_to_bytes(result))
            except Exception as e:
                print(f'Error in server: {e}')
                import traceback

                print(traceback.format_exc())
                error_response = {'status': 'error', 'message': str(e)}
                self.socket.send(self.convert_dict_to_bytes(error_response))
        
        # Cleanup when server stops
        self.executor.shutdown(wait=True)
        self.socket.close()
        self.context.term()

def main():
    """Main function for testing the server"""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='ZMQ Inference Server')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5555, help='Server port (default: 5555)')

    args = parser.parse_args()

    print(f"Starting ZMQ Inference Server on {args.host}:{args.port}")

    server = ZmqInferenceServer(args.host, args.port)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
