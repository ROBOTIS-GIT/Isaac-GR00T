#!/usr/bin/env python3
"""
GR00T N1.5 ZMQ Inference Server

This script runs a ZMQ server inside the GR00T container to handle
inference requests from Physical AI Server.

Usage:
    python scripts/zmq_inference_server.py --port 5556 --model_path nvidia/GR00T-N1.5-3B
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Add workspace to path
sys.path.insert(0, '/workspace')

try:
    import zmq
except ImportError:
    print("Error: pyzmq not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyzmq>=25.0.0"])
    import zmq

# Import GR00T modules
try:
    from gr00t.model import Gr00tPolicy
    # Note: GR00T does not have a preprocess_observation utility function
    # We'll implement our own preprocessing based on the observation structure
except ImportError as e:
    print(f"Error importing GR00T modules: {e}")
    print("Make sure you're running inside the GR00T Docker container")
    sys.exit(1)


class GR00TInferenceServer:
    """
    ZMQ-based inference server for GR00T N1.5.
    
    Handles inference requests from Physical AI Server and returns actions.
    """
    
    def __init__(
        self,
        model_path: str,
        port: int = 5556,
        device: str = 'cuda',
        embodiment_tag: str = 'default',
    ):
        """
        Initialize GR00T inference server.
        
        Args:
            model_path: Path to GR00T model checkpoint
            port: ZMQ server port
            device: Device to run inference on ('cuda' or 'cpu')
            embodiment_tag: Embodiment tag for the robot
        """
        self.port = port
        self.device = device
        self.embodiment_tag = embodiment_tag
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.logger.info(f"Loading GR00T model from {model_path}...")
        self.model = self._load_model(model_path)
        self.logger.info("Model loaded successfully")
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.logger.info(f"ZMQ server listening on tcp://*:{port}")
        
        # State
        self.request_count = 0
        self.running = False
    
    def _load_model(self, model_path: str) -> Gr00tPolicy:
        """Load GR00T policy model."""
        try:
            # Load model
            policy = Gr00tPolicy.from_pretrained(
                model_path,
                device=self.device,
                embodiment_tag=self.embodiment_tag,
            )
            policy.eval()
            
            return policy
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _handle_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle inference request.
        
        Args:
            data: Request data containing observation
        
        Returns:
            Result dictionary with action
        """
        try:
            observation = data.get('observation', {})
            
            # Preprocess observation
            # Expected keys: 'image', 'robot_state', etc.
            processed_obs = self._preprocess_observation(observation)
            
            # Run inference
            with torch.no_grad():
                action = self.model.predict(processed_obs)
            
            # Convert to list
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            if isinstance(action, np.ndarray):
                action = action.tolist()
            
            return {
                'action': action,
                'timestamp': time.time(),
            }
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            raise
    
    def _preprocess_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess observation for GR00T model.
        
        Args:
            observation: Raw observation dictionary
        
        Returns:
            Preprocessed observation
        """
        processed = {}
        
        # Image preprocessing
        if 'image' in observation:
            image = observation['image']
            if isinstance(image, list):
                image = np.array(image)
            
            # Convert to torch tensor
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            
            # Normalize and reshape if needed
            # GR00T expects (B, C, H, W)
            if image.dim() == 3:  # (H, W, C)
                image = image.permute(2, 0, 1)  # (C, H, W)
            
            if image.dim() == 3:  # (C, H, W)
                image = image.unsqueeze(0)  # (1, C, H, W)
            
            processed['image'] = image.to(self.device)
        
        # Robot state preprocessing
        if 'robot_state' in observation:
            robot_state = observation['robot_state']
            if isinstance(robot_state, list):
                robot_state = np.array(robot_state)
            
            if isinstance(robot_state, np.ndarray):
                robot_state = torch.from_numpy(robot_state)
            
            if robot_state.dim() == 1:
                robot_state = robot_state.unsqueeze(0)  # (1, D)
            
            processed['robot_state'] = robot_state.to(self.device)
        
        # Language instruction (if provided)
        if 'instruction' in observation:
            processed['instruction'] = observation['instruction']
        
        return processed
    
    def _handle_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check request."""
        return {
            'status': 'ok',
            'model_loaded': self.model is not None,
            'device': str(self.device),
            'requests_processed': self.request_count,
        }
    
    def _handle_command(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom command."""
        command = data.get('command')
        params = data.get('params', {})
        
        if command == 'get_info':
            return {
                'model_type': 'GR00T-N1.5',
                'embodiment_tag': self.embodiment_tag,
                'device': str(self.device),
            }
        elif command == 'reset':
            # Reset any internal state if needed
            return {'status': 'reset_complete'}
        else:
            raise ValueError(f"Unknown command: {command}")
    
    def run(self):
        """Run the ZMQ server loop."""
        self.running = True
        self.logger.info("GR00T inference server started")
        
        try:
            while self.running:
                # Wait for request
                message = self.socket.recv_json()
                
                msg_type = message.get('type')
                data = message.get('data', {})
                request_id = message.get('request_id', 'unknown')
                
                self.logger.debug(f"Received {msg_type} request: {request_id}")
                
                # Handle request
                try:
                    if msg_type == 'inference':
                        result = self._handle_inference(data)
                        response = {
                            'status': 'success',
                            'result': result,
                        }
                    elif msg_type == 'health':
                        result = self._handle_health(data)
                        response = {
                            'status': 'success',
                            'result': result,
                        }
                    elif msg_type == 'command':
                        result = self._handle_command(data)
                        response = {
                            'status': 'success',
                            'result': result,
                        }
                    else:
                        response = {
                            'status': 'error',
                            'error': f'Unknown message type: {msg_type}',
                        }
                    
                    self.request_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error handling request: {e}")
                    response = {
                        'status': 'error',
                        'error': str(e),
                    }
                
                # Add metadata
                response['request_id'] = request_id
                response['timestamp'] = time.time()
                
                # Send response
                self.socket.send_json(response)
                self.logger.debug(f"Sent response: {request_id}")
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server."""
        self.running = False
        try:
            self.socket.close()
            self.context.term()
            self.logger.info("Server stopped")
        except Exception as e:
            self.logger.error(f"Error stopping server: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='GR00T N1.5 ZMQ Inference Server'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5556,
        help='ZMQ server port (default: 5556)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='nvidia/GR00T-N1.5-3B',
        help='Path to GR00T model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--embodiment_tag',
        type=str,
        default='default',
        help='Embodiment tag for the robot'
    )
    
    args = parser.parse_args()
    
    # Create and run server
    server = GR00TInferenceServer(
        model_path=args.model_path,
        port=args.port,
        device=args.device,
        embodiment_tag=args.embodiment_tag,
    )
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
