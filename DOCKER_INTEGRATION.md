# GR00T N1.5 Docker 통합 가이드

Physical AI Tools에서 GR00T N1.5를 Docker로 실행하고 ZMQ로 통신하는 방법입니다.

## 🎯 목표

기존 CLI 방식:
```bash
# Build
docker build -t isaac-gr00t:latest .

# Run
docker run --gpus all --rm -it \
  --network host \
  --shm-size=64g \
  -v "$PWD":/workspace \
  -v /root/ext_data_storage/gr00t_n1_5_data:/workspace/checkpoints \
  -w /workspace \
  isaac-gr00t:latest
```

**→ Docker Manager + ZMQ로 자동화**

## 📋 변경 사항

### 1. Dockerfile 수정

```dockerfile
# pyzmq 추가
RUN pip install pyzmq>=25.0.0

# ZMQ 서버 스크립트 실행 권한
RUN chmod +x /workspace/scripts/zmq_inference_server.py

# 기본 명령어: ZMQ 서버 시작
CMD ["python", "scripts/zmq_inference_server.py", "--port", "5556"]
```

### 2. ZMQ Inference Server 추가

- 위치: `Isaac-GR00T/scripts/zmq_inference_server.py`
- 기능:
  - GR00T 모델 로드
  - ZMQ 서버 실행 (포트 5556)
  - Inference 요청 처리
  - Health check 지원

## 🚀 사용 방법

### 방법 1: 테스트 스크립트 사용 (추천)

```bash
cd /home/dongyun/ros2_ws/src/physical_ai_tools/physical_ai_server/physical_ai_server/docker_manager

# Docker 이미지 빌드 + 테스트
python test_groot_docker.py --build --test-inference

# 이미 빌드된 이미지로 테스트만
python test_groot_docker.py --test-inference

# Inference 테스트 생략
python test_groot_docker.py --no-inference
```

### 방법 2: Python 코드에서 직접 사용

```python
from physical_ai_server.docker_manager import (
    DockerManager,
    FrameworkType,
    ZMQClient,
)
from pathlib import Path

# 초기화
manager = DockerManager()

# GR00T 컨테이너 생성
container_config = {
    'shm_size': '64g',
    'network_mode': 'host',
}

container_id = manager.create_container(
    framework=FrameworkType.GROOT_N15,
    config=container_config,
    gpu_ids=None,  # 모든 GPU 사용
    api_port=5556,
)

# 컨테이너 시작
manager.start_container(container_id)

# ZMQ 서버 시작
process_config = {
    'port': 5556,
    'model_path': 'nvidia/GR00T-N1.5-3B',
    'embodiment_tag': 'default',
}

manager.start_framework_process(
    container_id,
    FrameworkType.GROOT_N15,
    process_config
)

# ZMQ 클라이언트로 통신
client = ZMQClient('tcp://localhost:5556')

# Inference 실행
observation = {
    'image': [...],  # 이미지 데이터
    'robot_state': [...],  # 로봇 상태
}
response = client.inference(observation)
action = response['result']['action']

# 정리
client.close()
manager.stop_container(container_id)
```

### 방법 3: Physical AI Server 통합

```python
# physical_ai_server.py
class PhysicalAIServer(Node):
    def __init__(self):
        super().__init__('physical_ai_server')
        
        self.docker_manager = DockerManager()
        self.zmq_clients = ZMQClientPool()
    
    def start_groot(self):
        """GR00T 시작"""
        # 컨테이너 생성
        container_id = self.docker_manager.create_container(
            framework=FrameworkType.GROOT_N15,
            config={'shm_size': '64g', 'network_mode': 'host'},
            gpu_ids=[0],
        )
        
        # 시작
        self.docker_manager.start_container(container_id)
        
        # ZMQ 서버 프로세스 시작
        self.docker_manager.start_framework_process(
            container_id,
            FrameworkType.GROOT_N15,
            {'port': 5556, 'model_path': 'nvidia/GR00T-N1.5-3B'}
        )
        
        # ZMQ 클라이언트 연결
        self.zmq_clients.add_client('groot', 'tcp://localhost:5556')
        
        self.get_logger().info("GR00T started")
    
    def run_groot_inference(self, observation):
        """GR00T 추론 실행"""
        response = self.zmq_clients.inference('groot', observation)
        return response['result']['action']
```

## 🔧 설정

### 환경 변수

```bash
# GPU 사용
export NVIDIA_VISIBLE_DEVICES=0,1

# 체크포인트 경로
export GROOT_CHECKPOINT_PATH=/root/ext_data_storage/gr00t_n1_5_data
```

### 볼륨 매핑

```python
# Docker Manager가 자동으로 매핑
~/.cache/physical_ai_tools/shared_data  → /workspace/shared_data
~/.cache/physical_ai_tools/groot_n15    → /workspace/groot_n15_data
```

### 추가 볼륨 (checkpoint)

```python
from docker.types import Mount

volumes = [
    Mount(
        target='/workspace/checkpoints',
        source='/root/ext_data_storage/gr00t_n1_5_data',
        type='bind',
    )
]

container_id = manager.create_container(
    framework=FrameworkType.GROOT_N15,
    config={'mounts': volumes},
)
```

## 📊 통신 플로우

```
1. Physical AI Server 시작
   ↓
2. Docker Manager가 GR00T 컨테이너 생성
   ↓
3. 컨테이너 내부에서 ZMQ 서버 시작
   - 포트: 5556
   - GR00T 모델 로드
   ↓
4. Physical AI Server가 ZMQ Client 생성
   ↓
5. 로봇에서 관측 데이터 수신 (ROS2)
   ↓
6. ZMQ로 GR00T에 inference 요청
   {
     "type": "inference",
     "data": {
       "observation": {
         "image": [...],
         "robot_state": [...]
       }
     }
   }
   ↓
7. GR00T이 추론 실행
   ↓
8. ZMQ로 응답
   {
     "status": "success",
     "result": {
       "action": [0.1, 0.2, ...]
     }
   }
   ↓
9. Physical AI Server가 액션을 로봇에 전송 (ROS2)
```

## 🧪 테스트

### 1. Docker 이미지 빌드 테스트

```bash
cd /home/dongyun/ros2_ws/src/physical_ai_tools/Isaac-GR00T
docker build -t robotis/groot_n15:latest .
```

### 2. ZMQ 서버 단독 테스트

```bash
# 컨테이너 실행
docker run --gpus all -it \
  --network host \
  --shm-size=64g \
  -v "$PWD":/workspace \
  -v /root/ext_data_storage/gr00t_n1_5_data:/workspace/checkpoints \
  robotis/groot_n15:latest

# 컨테이너 내부에서
python scripts/zmq_inference_server.py --port 5556 --model_path nvidia/GR00T-N1.5-3B
```

### 3. ZMQ 클라이언트 테스트

```bash
# 다른 터미널에서
python -c "
from physical_ai_server.docker_manager import ZMQClient
import numpy as np

client = ZMQClient('tcp://localhost:5556', timeout=30000)

# Health check
print('Health:', client.health_check())

# Inference
observation = {
    'image': np.random.rand(224, 224, 3).tolist(),
    'robot_state': [0.1, 0.2, 0.3, 0.4, 0.5],
}
response = client.inference(observation)
print('Action:', response['result']['action'][:5])

client.close()
"
```

### 4. 통합 테스트

```bash
cd /home/dongyun/ros2_ws/src/physical_ai_tools/physical_ai_server/physical_ai_server/docker_manager
python test_groot_docker.py --build --test-inference
```

## 🐛 문제 해결

### 1. Docker 빌드 실패

```bash
# CUDA 버전 확인
nvidia-smi

# Docker NVIDIA runtime 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 2. ZMQ 서버 시작 실패

```bash
# 컨테이너 로그 확인
docker logs <container_id>

# 컨테이너 내부 접속
docker exec -it <container_id> bash

# ZMQ 서버 로그
tail -f /tmp/groot_n15.log
```

### 3. 모델 로딩 실패

```bash
# Hugging Face 캐시 확인
ls -lh ~/.cache/huggingface

# 모델 다운로드
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('nvidia/GR00T-N1.5-3B')
"
```

### 4. GPU 메모리 부족

```python
# 메모리 제한 설정
container_id = manager.create_container(
    framework=FrameworkType.GROOT_N15,
    memory_limit='32g',  # 메모리 제한
)
```

### 5. 포트 충돌

```bash
# 포트 사용 확인
sudo netstat -tulpn | grep 5556

# 프로세스 종료
sudo kill <PID>
```

## 📝 CLI vs Docker Manager 비교

| 항목 | CLI 방식 | Docker Manager 방식 |
|------|----------|-------------------|
| **빌드** | `docker build -t isaac-gr00t:latest .` | `manager.build_image(...)` |
| **실행** | 긴 docker run 명령어 | `manager.create_container(...)` |
| **GPU** | `--gpus all` | `gpu_ids=[0, 1]` |
| **볼륨** | `-v` 여러 번 | `config={'mounts': [...]}` |
| **통신** | 수동으로 설정 | ZMQ 자동 설정 |
| **모니터링** | `docker stats` | `manager.monitor_resources()` |
| **정리** | `docker stop/rm` | `manager.stop_container()` |

## 🎓 다음 단계

1. **Fine-tuning 지원**
   - 학습 요청 처리
   - 체크포인트 저장/로드

2. **Multi-GPU 지원**
   - GPU 할당 전략
   - 분산 학습

3. **모델 최적화**
   - TensorRT 지원
   - ONNX 변환

4. **자동 복구**
   - 컨테이너 재시작
   - Health check 기반 자동 복구

## 📞 지원

문제가 있으시면:
1. 로그 확인: `docker logs <container_id>`
2. 디버그 모드: `--log-level DEBUG`
3. GitHub Issues 등록

---

**GR00T N1.5가 이제 Physical AI Tools의 일부입니다!** 🚀
