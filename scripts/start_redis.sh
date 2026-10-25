#!/usr/bin/env bash
# Start a local Redis (message broker for configs/test5_yolo11_triton_redis.txt) on port 6379.
# Read messages:  docker exec redis-ds redis-cli XLEN ds-yolo11
#                 docker exec redis-ds redis-cli --raw XREVRANGE ds-yolo11 + - COUNT 1
set -e
NAME=redis-ds
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME -p 6379:6379 redis:7-alpine >/dev/null
for i in $(seq 20); do
  docker exec $NAME redis-cli ping 2>/dev/null | grep -q PONG && { echo "Redis READY on localhost:6379 (stream key: ds-yolo11)"; exit 0; }
  sleep 1
done
echo "Redis did not start"; exit 1
