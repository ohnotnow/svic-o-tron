#!/bin/bash
set -e

# ensure we have one parameter
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <name-of-bot>"
  echo "Example: $0 gepetto"
  echo "(There should be a matching .env.gepetto, for instance)"
  exit 1
fi
BOT_NAME=$1

# Check if the container is already running
if docker ps --format '{{.Names}}' | grep -q "^${BOT_NAME}$"; then
  echo "${BOT_NAME} Container is already running"
  exit 1
fi

docker build -t ${BOT_NAME} .

# put your various environment variables in a file named .env
docker run --restart=on-failure --env-file=.env.${BOT_NAME} -v $(pwd)/stats.json:/app/stats.json ${BOT_NAME}
