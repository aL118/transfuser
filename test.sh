export CARLA_ROOT="/fs/nexus-scratch/aliu1237/transfuser/carla"
export WORK_DIR="/fs/nexus-scratch/aliu1237/transfuser"

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6/longest6.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/temp/transfuser_neat.json
export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/submission_agent.py
export TEAM_CONFIG=${WORK_DIR}/model_ckpt/models_2022/transfuser
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
# /fs/nexus-scratch/aliu1237/transfuser/results/save_results/transfuser_neat.json
python ${WORK_DIR}/tools/result_parser.py --xml ${WORK_DIR}/leaderboard/data/neat/eval_routes.xml \
 --results /fs/nexus-scratch/aliu1237/transfuser/results/temp --save_dir /fs/nexus-scratch/aliu1237/transfuser/results/save_results \
 --town_maps ${WORK_DIR}/leaderboard/data/town_maps_xodr_neat
