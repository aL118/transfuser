export CARLA_ROOT="/fs/nexus-projects/sim2real/lyzheng/transfuser/carla"
export WORK_DIR="/fs/nexus-projects/sim2real/lyzheng/transfuser"

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/leaderboard/data/neat/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/neat/eval_routes.xml
export REPETITIONS=1

export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/transfuser_neat.json
export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/submission_agent.py

export TEAM_CONFIG=${WORK_DIR}/models/transfuser

export DEBUG_CHALLENGE=0
export RESUME=0
export DATAGEN=0
export PORT=2000

export SAVE_PATH="$WORK_DIR/debug_output" # uncomment for debug output

# Cleanup function with progress
cleanup() {
    echo ""
    echo "🛑 Script interrupted! Performing cleanup..."
    
    # Kill evaluator if running
    if [[ -n $EVALUATOR_PID ]]; then
        echo "📊 Stopping evaluator (PID: $EVALUATOR_PID)..."
        kill -TERM $EVALUATOR_PID 2>/dev/null
    fi
    
    # Kill CARLA
    if [[ -n $CARLA_PID ]]; then
        echo "🚗 Stopping CARLA server (PID: $CARLA_PID)..."
        kill -TERM $CARLA_PID 2>/dev/null
        sleep 3
        kill -KILL $CARLA_PID 2>/dev/null
    fi
    
    # Nuclear option - kill all CARLA processes
    echo "🔥 Killing all CARLA processes..."
    pkill -f -KILL CarlaUE4 2>/dev/null
    
    # Verify cleanup
    if pgrep -f CarlaUE4 > /dev/null; then
        echo "⚠️  Some CARLA processes may still be running"
    else
        echo "✅ All CARLA processes stopped"
    fi
    
    exit 0
}

# Set trap
trap cleanup SIGINT SIGTERM

echo "🚀 Starting CARLA evaluation (Press Ctrl+C to stop cleanly)"

# Start CARLA
DISPLAY= ${CARLA_ROOT}/CarlaUE4.sh -opengl -carla-port=${PORT} -fps=20 -nosound > $WORK_DIR/logs/carla.log 2>&1 & 
CARLA_PID=$!
echo "CARLA started with PID: $CARLA_PID"

sleep 15

# Run evaluator in background to capture its PID
python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--port=${PORT} \
--resume=${RESUME} > $WORK_DIR/logs/evaluation.log 2>&1 &

EVALUATOR_PID=$!
echo "Evaluator started with PID: $EVALUATOR_PID"

# Wait for evaluator to complete
wait $EVALUATOR_PID

# Normal cleanup
cleanup