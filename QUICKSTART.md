# 🚀 GR00T N1.5 Docker 빠른 시작

Physical AI Tools에서 GR00T N1.5를 Docker로 실행하는 가장 빠른 방법입니다.

## 📋 준비사항

```bash
# 1. NVIDIA Docker 설치 확인
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 2. Python 패키지 설치
pip install docker>=7.0.0 pyzmq>=25.0.0
```

## ⚡ 빠른 시작 (3가지 방법)

### 방법 1: 쉘 스크립트 (가장 빠름) ✨

```bash
cd /home/dongyun/ros2_ws/src/physical_ai_tools/Isaac-GR00T

# 1. Docker 이미지 빌드
./docker_quickstart.sh build

# 2. 컨테이너 실행
./docker_quickstart.sh run-cli

# 3. 상태 확인
./docker_quickstart.sh status

# 4. ZMQ 통신 테스트
./docker_quickstart.sh test

# 5. 로그 확인
./docker_quickstart.sh logs

# 6. 컨테이너 접속
./docker_quickstart.sh enter

# 7. 정리
./docker_quickstart.sh stop
./docker_quickstart.sh remove
```

### 방법 2: Python 테스트 스크립트

```bash
cd /home/dongyun/ros2_ws/src/physical_ai_tools/physical_ai_server/physical_ai_server/docker_manager

# 빌드 + 실행 + 테스트
python test_groot_docker.py --build --test-inference

# 이미 빌드된 경우
python test_groot_docker.py --test-inference
```

### 방법 3: 전통적인 Docker CLI

```bash
cd /home/dongyun/ros2_ws/src/physical_ai_tools/Isaac-GR00T

# 빌드
docker build -t robotis/groot_n15:latest .

# 실행
docker run --gpus all -d \
  --name physical_ai_groot \
  --network host \
  --shm-size=64g \
  -v "$PWD":/workspace \
  -v /root/ext_data_storage/gr00t_n1_5_data:/workspace/checkpoints \
  -w /workspace \
  robotis/groot_n15:latest \
  python scripts/zmq_inference_server.py --port 5556

# 테스트
python -c "
from physical_ai_server.docker_manager import ZMQClient
client = ZMQClient('tcp://localhost:5556')
print('Health:', client.health_check())
"
```

## 🧪 ZMQ 통신 테스트

### 1. Health Check
```bash
python -c "
from physical_ai_server.docker_manager import ZMQClient

client = ZMQClient('tcp://localhost:5556', timeout=5000)
if client.health_check():
    print('✓ GR00T is healthy')
else:
    print('✗ GR00T is not responding')
client.close()
"
```

### 2. Inference Test
```bash
python -c "
from physical_ai_server.docker_manager import ZMQClient
import numpy as np

client = ZMQClient('tcp://localhost:5556', timeout=30000)

# 더미 관측 데이터
observation = {
    'image': np.random.rand(224, 224, 3).tolist(),
    'robot_state': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'instruction': 'Pick up the coffee bottle',
}

# 추론 요청
response = client.inference(observation)

if response and response['status'] == 'success':
    action = response['result']['action']
    print(f'✓ Action received: {action[:5]}...')
else:
    print(f'✗ Inference failed: {response}')

client.close()
"
```

## 📖 Docker Manager로 사용하기

```python
from physical_ai_server.docker_manager import (
    DockerManager,
    FrameworkType,
    ZMQClient,
)

# 초기화
manager = DockerManager()

# 컨테이너 생성
container_id = manager.create_container(
    framework=FrameworkType.GROOT_N15,
    config={'shm_size': '64g', 'network_mode': 'host'},
    gpu_ids=None,  # 모든 GPU
    api_port=5556,
)

# 시작
manager.start_container(container_id)

# ZMQ 서버 시작
manager.start_framework_process(
    container_id,
    FrameworkType.GROOT_N15,
    {
        'port': 5556,
        'model_path': 'nvidia/GR00T-N1.5-3B',
        'embodiment_tag': 'default',
    }
)

# ZMQ 통신
client = ZMQClient('tcp://localhost:5556')
response = client.inference({'image': [...], 'robot_state': [...]})
action = response['result']['action']

# 정리
client.close()
manager.stop_container(container_id)
```

## 🔧 설정 커스터마이징

### 다른 모델 사용
```bash
# Fine-tuned 모델 사용
docker run ... \
  python scripts/zmq_inference_server.py \
    --port 5556 \
    --model_path /workspace/checkpoints/my_finetuned_model
```

### GPU 지정
```bash
# 특정 GPU만 사용
docker run ... \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  ...
```

### 메모리 제한
```bash
# 메모리 제한 설정
docker run ... \
  --memory="32g" \
  ...
```

## 🐛 문제 해결

### 컨테이너가 시작되지 않음
```bash
# 로그 확인
./docker_quickstart.sh logs

# 또는
docker logs physical_ai_groot
```

### ZMQ 연결 실패
```bash
# 포트 확인
netstat -tulpn | grep 5556

# 컨테이너 상태 확인
./docker_quickstart.sh status
```

### GPU 인식 안 됨
```bash
# NVIDIA Docker runtime 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 컨테이너 내부에서 확인
./docker_quickstart.sh enter
python -c "import torch; print(torch.cuda.is_available())"
```

### 모델 로딩 실패
```bash
# 컨테이너 접속
./docker_quickstart.sh enter

# 모델 다운로드
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('nvidia/GR00T-N1.5-3B')
"
```

## 📊 성능 모니터링

```bash
# 리소스 사용량
docker stats physical_ai_groot

# 또는 Docker Manager로
python -c "
from physical_ai_server.docker_manager import DockerManager

manager = DockerManager()
container_id = 'physical_ai_groot'
resources = manager.monitor_resources(container_id)

print(f'CPU: {resources[\"cpu_percent\"]}%')
print(f'Memory: {resources[\"memory_percent\"]}%')
"
```

## 📁 파일 구조

```
Isaac-GR00T/
├── Dockerfile                      # ✅ ZMQ 지원 추가됨
├── docker_quickstart.sh            # ✅ 쉘 스크립트
├── DOCKER_INTEGRATION.md           # ✅ 상세 가이드
├── QUICKSTART.md                   # ✅ 이 파일
└── scripts/
    └── zmq_inference_server.py     # ✅ ZMQ 서버

physical_ai_server/
└── docker_manager/
    ├── docker_manager.py           # ✅ GR00T 지원
    ├── zmq_communication.py        # ✅ ZMQ 통신
    └── test_groot_docker.py        # ✅ 테스트 스크립트
```

## 🎓 다음 단계

1. **Fine-tuning**
   - 자신의 데이터로 학습
   - 체크포인트 관리

2. **Physical AI Server 통합**
   - ROS2와 통합
   - 로봇 제어

3. **멀티 프레임워크**
   - LeRobot + GR00T 동시 실행
   - 프레임워크 간 비교

## 📞 지원

- 📖 상세 가이드: `DOCKER_INTEGRATION.md`
- 🐛 문제 발생: GitHub Issues
- 💬 질문: kdy@robotis.com

---

**이제 GR00T N1.5를 Docker로 쉽게 사용할 수 있습니다!** 🎉
