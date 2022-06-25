
# 1. Build Docker Image

```bash
cd docker
./build-docker.sh
```

# 2. Start Inference Docker

```bash
cd docker
./run-inference-docker.sh
```
You can modify the API port in the `run-inference-docker.sh` script.

# 3. Send Request

Example:

```bash
curl -X POST http://localhost:6666/api/inference -H 'Content-Type: application/json' -d '{"query_context":["你好 我的名字叫屁贼", "我也是"]}'
```
