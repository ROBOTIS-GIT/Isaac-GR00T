#!/bin/bash
# GR00T N1.5 Docker Quick Start Script
# 이 스크립트는 GR00T Docker를 빠르게 시작하기 위한 헬퍼입니다.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GROOT_PATH="/home/dongyun/ros2_ws/src/physical_ai_tools/Isaac-GR00T"
CHECKPOINT_PATH="/root/ext_data_storage/gr00t_n1_5_data"
IMAGE_NAME="robotis/groot_n15:latest"
CONTAINER_NAME="physical_ai_groot"
ZMQ_PORT=5556

print_header() {
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function: Build Docker image
build_image() {
    print_header "Building GR00T Docker Image"
    
    if [ ! -d "$GROOT_PATH" ]; then
        print_error "GR00T directory not found: $GROOT_PATH"
        exit 1
    fi
    
    cd "$GROOT_PATH"
    print_info "Building from: $GROOT_PATH"
    
    docker build -t "$IMAGE_NAME" .
    
    print_success "Image built: $IMAGE_NAME"
}

# Function: Run container (CLI style)
run_cli() {
    print_header "Running GR00T Container (CLI Style)"
    
    print_info "Starting container: $CONTAINER_NAME"
    
    docker run --gpus all -d \
        --name "$CONTAINER_NAME" \
        --network host \
        --shm-size=64g \
        -v "$GROOT_PATH":/workspace \
        -v "$CHECKPOINT_PATH":/workspace/checkpoints \
        -w /workspace \
        "$IMAGE_NAME" \
        python scripts/zmq_inference_server.py --port "$ZMQ_PORT"
    
    print_success "Container started: $CONTAINER_NAME"
    print_info "ZMQ server running on port: $ZMQ_PORT"
}

# Function: Run with Docker Manager
run_manager() {
    print_header "Running GR00T with Docker Manager"
    
    cd /home/dongyun/ros2_ws/src/physical_ai_tools/physical_ai_server/physical_ai_server/docker_manager
    
    python test_groot_docker.py --test-inference
}

# Function: Stop container
stop_container() {
    print_header "Stopping GR00T Container"
    
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker stop "$CONTAINER_NAME"
        print_success "Container stopped"
    else
        print_info "Container not running"
    fi
}

# Function: Remove container
remove_container() {
    print_header "Removing GR00T Container"
    
    if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
        docker rm -f "$CONTAINER_NAME"
        print_success "Container removed"
    else
        print_info "Container does not exist"
    fi
}

# Function: Show logs
show_logs() {
    print_header "GR00T Container Logs"
    
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker logs -f "$CONTAINER_NAME"
    else
        print_error "Container not running"
    fi
}

# Function: Enter container
enter_container() {
    print_header "Entering GR00T Container"
    
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker exec -it "$CONTAINER_NAME" bash
    else
        print_error "Container not running"
    fi
}

# Function: Test ZMQ connection
test_zmq() {
    print_header "Testing ZMQ Connection"
    
    python3 << EOF
import sys
sys.path.insert(0, '/home/dongyun/ros2_ws/src/physical_ai_tools/physical_ai_server')

from physical_ai_server.docker_manager import ZMQClient
import numpy as np

print("Connecting to ZMQ server...")
client = ZMQClient('tcp://localhost:$ZMQ_PORT', timeout=5000)

print("Health check...")
if client.health_check():
    print("✓ Server is healthy")
    
    print("\nTesting inference...")
    observation = {
        'image': np.random.rand(224, 224, 3).tolist(),
        'robot_state': [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    
    response = client.inference(observation)
    if response and response.get('status') == 'success':
        action = response['result']['action']
        print(f"✓ Inference successful")
        print(f"  Action (first 5): {action[:5]}")
    else:
        print(f"✗ Inference failed: {response}")
else:
    print("✗ Server is not responding")

client.close()
EOF
}

# Function: Show status
show_status() {
    print_header "GR00T Container Status"
    
    # Check if container exists
    if docker ps -aq -f name="$CONTAINER_NAME" | grep -q .; then
        # Check if running
        if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
            print_success "Container is RUNNING"
            
            # Show container info
            echo ""
            docker ps -f name="$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            
            # Show resource usage
            echo ""
            print_info "Resource Usage:"
            docker stats --no-stream "$CONTAINER_NAME"
        else
            print_info "Container EXISTS but STOPPED"
        fi
    else
        print_info "Container does not exist"
    fi
    
    # Check ZMQ connection
    echo ""
    print_info "Checking ZMQ connection..."
    if nc -z localhost "$ZMQ_PORT" 2>/dev/null; then
        print_success "ZMQ port $ZMQ_PORT is OPEN"
    else
        print_info "ZMQ port $ZMQ_PORT is CLOSED"
    fi
}

# Function: Show help
show_help() {
    cat << EOF
GR00T N1.5 Docker Quick Start

Usage: $0 [COMMAND]

Commands:
  build           Build GR00T Docker image
  run-cli         Run container (traditional CLI style)
  run-manager     Run with Docker Manager (Python)
  stop            Stop container
  remove          Remove container
  logs            Show container logs
  enter           Enter container shell
  test            Test ZMQ connection
  status          Show container status
  help            Show this help message

Examples:
  # First time setup
  $0 build
  $0 run-cli
  
  # Check status
  $0 status
  
  # Test communication
  $0 test
  
  # View logs
  $0 logs
  
  # Stop and cleanup
  $0 stop
  $0 remove

Configuration:
  Image:      $IMAGE_NAME
  Container:  $CONTAINER_NAME
  ZMQ Port:   $ZMQ_PORT
  GR00T Path: $GROOT_PATH
  Checkpoint: $CHECKPOINT_PATH

EOF
}

# Main script
case "$1" in
    build)
        build_image
        ;;
    run-cli)
        run_cli
        ;;
    run-manager)
        run_manager
        ;;
    stop)
        stop_container
        ;;
    remove)
        remove_container
        ;;
    logs)
        show_logs
        ;;
    enter)
        enter_container
        ;;
    test)
        test_zmq
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
