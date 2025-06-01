#!/bin/bash
# QuantoniumOS Redis Service Startup Script
# Starts local Redis instance for rate limiting and security caching

REDIS_PID_FILE="/tmp/redis.pid"
REDIS_LOG_FILE="/tmp/redis.log"
REDIS_CONF_FILE="/tmp/redis.conf"

# Create basic Redis configuration
cat > $REDIS_CONF_FILE << EOF
daemonize yes
pidfile $REDIS_PID_FILE
logfile $REDIS_LOG_FILE
port 6379
bind 127.0.0.1
timeout 300
tcp-keepalive 60
maxmemory 50mb
maxmemory-policy allkeys-lru
save ""
appendonly no
EOF

# Start Redis if not already running
if [ ! -f $REDIS_PID_FILE ] || ! kill -0 $(cat $REDIS_PID_FILE) 2>/dev/null; then
    echo "Starting local Redis server..."
    redis-server $REDIS_CONF_FILE
    sleep 2
    if [ -f $REDIS_PID_FILE ] && kill -0 $(cat $REDIS_PID_FILE) 2>/dev/null; then
        echo "Redis started successfully on 127.0.0.1:6379"
    else
        echo "Failed to start Redis"
        exit 1
    fi
else
    echo "Redis already running"
fi