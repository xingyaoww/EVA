docker run \
    -d \
    -p 6666:6666 \
    --gpus 0 \
    -v ${PWD}:/home/coai/EVA \
    --name EVA2-inference-server \
    eva:1.5.cu113 /bin/bash /home/coai/EVA/src/scripts/eva_inference_server_no_beam.sh

# Example Request:
# curl -X POST http://i.tech.corgi.plus:6666/api/inference -H 'Content-Type: application/json' -d '{"query_context":["你好 我的名字叫屁贼", "我也是"]}'
