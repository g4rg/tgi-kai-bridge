#!/bin/bash

echo TGI Horde Worker

fatal () {
    echo "Error: $1"
    exit 1
}

[ -z "$HORDE_WORKER_NAME" ] && fatal "missing environment var: HORDE_WORKER_NAME"
[ -z "$HORDE_API_KEY" ] && fatal "missing environment var: HORDE_API_KEY"
[ -z "$HORDE_MODEL" ] && fatal "missing environment var: HORDE_MODEL"

if [ -z "$HORDE_MAX_LENGTH" ] ; then
    echo "Warning: HORDE_MAX_LENGTH not set, defaulting to 80"
    HORDE_MAX_LENGTH=80
fi
if [ -z "$HORDE_MAX_CONTEXT_LENGTH" ] ; then
    echo "Warning: HORDE_MAX_CONTEXT_LENGTH not set, defaulting to 1024"
    HORDE_MAX_CONTEXT_LENGTH=1024
fi
if [ -z "$HORDE_MAX_THREADS" ] ; then
    echo "Warning: HORDE_MAX_THREADS not set, defaulting to 1"
    HORDE_MAX_THREADS=1
fi

cat <<EOF > ./AI-Horde-Worker/bridgeData.yaml
worker_name: "$HORDE_WORKER_NAME"
api_key: "$HORDE_API_KEY"
kai_url: "http://127.0.0.1:5000"
max_length: $HORDE_MAX_LENGTH
max_context_length: $HORDE_MAX_CONTEXT_LENGTH
max_threads: $HORDE_MAX_THREADS
queue_size: 1
EOF

echo "starting TGI"
text-generation-launcher --model-id "$HORDE_MODEL" --max-total-tokens "$HORDE_MAX_CONTEXT_LENGTH" --max-input-length "$(($HORDE_MAX_CONTEXT_LENGTH-1))" &
PID_TGI=$!
while :
do
    [ -d "/proc/${PID_TGI}" ] || fatal "failed to start TGI" # check process
    nc -z 127.0.0.1 3000 && break # check API
    sleep 1
done
echo "TGI launch success"

echo "starting tgi-kai-bridge"
( cd tgi-kai-bridge && exec python main.py ) &
PID_BRIDGE=$!
while :
do
    [ -d "/proc/${PID_BRIDGE}" ] || fatal "failed to start tgi-kai-bridge" # check process
    nc -z 127.0.0.1 5000 && break # check API
    sleep 1
done
echo "tgi-kai-bridge launch success"

echo "starting AI Horde Worker"
cd AI-Horde-Worker && exec python -s bridge_scribe.py
