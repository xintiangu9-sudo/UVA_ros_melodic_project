#!/bin/bash
# start_ros.sh
# 全自动 ROS 启动器 + Python 3.10 虚拟环境

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <launch|run> <package file_or_node>"
    echo "示例1 (launch): $0 launch swarm_rescue dual_drone_env.launch"
    echo "示例2 (run):    $0 run swarm_rescue ros_bridge_node.py"
    exit 1
fi

MODE=$1
shift  # 剩余参数为文件或节点

# 激活虚拟环境
VENV_DIR=~/UVA/src/swarm_rescue/venv310
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "虚拟环境 $VENV_DIR 不存在！"
    exit 1
fi

# 激活 ROS workspace
ROS_WS=~/UVA/src/swarm_rescue
if [ -f "$ROS_WS/devel/setup.bash" ]; then
    source "$ROS_WS/devel/setup.bash"
else
    echo "ROS workspace setup.bash 不存在！"
    exit 1
fi

# 确保虚拟环境包可用
export PYTHONPATH="$VENV_DIR/lib/python3.10/site-packages:$PYTHONPATH"

echo "虚拟环境和 ROS 已激活！开始执行命令..."

case "$MODE" in
    launch)
        echo "运行 roslaunch: $@"
        roslaunch "$@"
        ;;
    run)
        echo "运行 rosrun: $@"
        rosrun "$@"
        ;;
    *)
        echo "未知模式：$MODE，请使用 launch 或 run"
        exit 1
        ;;
esac