#!/bin/bash

# Parallel ZO vs FO Parameter Sweep Script
# 支持并行运行和GPU选择的参数扫描脚本

# 不要使用 set -e，因为我们需要手动处理错误
# set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 全局变量：存储所有子进程 PID
declare -a ALL_CHILD_PIDS=()
MAIN_PID=$$
PID_FILE=""
CLEANUP_DONE=false

# 默认配置参数
MODES=("FO" "Instruct") # 可选: FO, ZO, Calibrate, Instruct
SCOPES=("full")
BATCH_SIZES=(8)
BLOCK_SIZES=(512)  # 序列长度 (可选: 64, 128, 256, 512, 1024)
QUERY_BUDGETS=(8)
BP_INTERVALS=(1)
INSTRUCT_COSINE_TARGETS=(0.01)
INSTRUCT_NOISE_SCALES=(10.0)
LEARNING_RATES_ZO=(1e-3)
OPTIMIZERS=("mudamw")  # 可选: sgd, adam, mudamw
EPOCHS=10
LOG_INTERVAL=10

# 学习率调度器配置 (Learning Rate Scheduler Configuration)
USE_LR_SCHEDULER=true  # 是否启用余弦退火学习率调度器
WARMUP_STEPS=300       # 预热步数
MIN_LR=1e-6           # 最小学习率

# 梯度累积配置 (Gradient Accumulation Configuration)
# 仅适用于FO模式。有效batch size = batch_size * gradient_accumulation_steps
GRADIENT_ACCUMULATION_STEPS=8  # 梯度累积步数，1表示不使用梯度累积

LOGS_ROOT="logs"

# 模型配置 (Model Configuration)
# 备选: 20M (超小型，快速实验), 200M (中小型，类似GPT-2 Small), 500M (中型), 1B (大型)
MODEL_SIZES=("20M")  # 默认使用200M模型，可以是数组如: ("20M" "200M" "500M" "1B")

# 数据集配置 (Dataset Configuration)
# 备选数据集:
#   - cosmopedia-100k: 高质量合成教育数据，100k样本，快速实验 (推荐用于快速测试)
#   - cosmopedia: Cosmopedia完整版，30M+样本，高质量
#   - wikitext-103: 维基百科文本，经典预训练数据集
#   - openwebtext: 开源WebText，8M+网页文档，接近真实分布
#   - c4: 超大规模清洗网页数据，365M文档，适合大规模预训练
#   - tinystories: 简单故事数据集，适合小模型调试
#   - pile-subset: The Pile无版权子集，多样化高质量数据
#   - fineweb: FineWeb完整版，15T tokens，主流高质量预训练数据 (推荐用于正式训练)
#   - fineweb-edu: FineWeb教育子集，1.3T tokens，高质量推荐
#   - fineweb-edu-10bt: FineWeb-Edu 10BT采样，适合快速实验
DATASET="fineweb-edu-10bt"  # 默认使用cosmopedia-100k (快速测试推荐fineweb-edu-10bt)

# 数据集最大样本数 (Dataset Max Samples)
# 设置为空字符串表示使用数据集的推荐值
# 建议值参考:
#   cosmopedia-100k(20000), cosmopedia(100000), openwebtext(50000), c4(100000)
#   fineweb(100000), fineweb-edu(50000), fineweb-edu-10bt(30000)
MAX_SAMPLES=""  # 留空使用推荐值，或指定具体数字如: 20000

# BP数据集配置 (BP Dataset Configuration for Calibrate/Instruct modes)
# 用于Calibrate/Instruct模式中BP梯度计算的数据集
# 留空表示使用与主训练相同的数据集
BP_DATASET="fineweb-edu-10bt"  # 留空使用主数据集，或指定不同的数据集如: "cosmopedia-100k"
BP_MAX_SAMPLES=""  # 留空使用推荐值，或指定具体数字

# 并行配置
MAX_PARALLEL_JOBS=32 # 最大并行任务数
GPU_IDS="2"           # GPU ID列表，空表示自动检测

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --modes)
            IFS=',' read -ra MODES <<< "$2"
            shift 2
            ;;
        --scopes)
            IFS=',' read -ra SCOPES <<< "$2"
            shift 2
            ;;
        --batch-sizes)
            IFS=',' read -ra BATCH_SIZES <<< "$2"
            shift 2
            ;;
        --block-sizes)
            IFS=',' read -ra BLOCK_SIZES <<< "$2"
            shift 2
            ;;
        --query-budgets)
            IFS=',' read -ra QUERY_BUDGETS <<< "$2"
            shift 2
            ;;
        --bp-intervals)
            IFS=',' read -ra BP_INTERVALS <<< "$2"
            shift 2
            ;;
        --learning-rates)
            IFS=',' read -ra LEARNING_RATES_ZO <<< "$2"
            shift 2
            ;;
        --optimizers)
            IFS=',' read -ra OPTIMIZERS <<< "$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --model-size|--model-sizes)
            IFS=',' read -ra MODEL_SIZES <<< "$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --bp-dataset)
            BP_DATASET="$2"
            shift 2
            ;;
        --bp-max-samples)
            BP_MAX_SAMPLES="$2"
            shift 2
            ;;
        --instruct-cosine-targets)
            IFS=',' read -ra INSTRUCT_COSINE_TARGETS <<< "$2"
            shift 2
            ;;
        --instruct-noise-scales)
            IFS=',' read -ra INSTRUCT_NOISE_SCALES <<< "$2"
            shift 2
            ;;
        --use-lr-scheduler)
            USE_LR_SCHEDULER=true
            shift 1
            ;;
        --no-lr-scheduler)
            USE_LR_SCHEDULER=false
            shift 1
            ;;
        --warmup-steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --min-lr)
            MIN_LR="$2"
            shift 2
            ;;
        --gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --parallel N         最大并行任务数 (默认: 32)"
            echo "  --gpus '0,1,2'      指定GPU ID列表，支持逗号或空格分隔 (默认: 自动检测)"
            echo "  --modes 'FO,ZO,Calibrate,Instruct'     优化方法 (默认: ZO)"
            echo "  --scopes 'reduced,full' 训练范围 (默认: full)"
            echo "  --batch-sizes '1,2,4' 批次大小 (默认: 2)"
            echo "  --block-sizes '64,128,256' 序列长度/块大小 (默认: 128)"
            echo "  --query-budgets '1,2,4,8' Query budget (默认: 1,2,4,...,1024)"
            echo "  --bp-intervals '1,2,5,10'  Calibrate/Instruct 模式的BP间隔 (默认: 1,2,5,10)"
            echo "  --learning-rates '1e-3,1e-4' 学习率 (默认: 1e-3)"
            echo "  --optimizers 'sgd,adam,mudamw' 优化器 (默认: mudamw)"
            echo "  --epochs N           训练轮数 (默认: 10)"
            echo "  --log-interval N     日志间隔 (默认: 10)"
            echo "  --model-size(s) SIZE 模型大小，支持多个: '20M,200M,500M,1B' (默认: 200M)"
            echo "  --dataset NAME       数据集名称 (默认: cosmopedia-100k)"
            echo "                       可选: cosmopedia-100k, cosmopedia, wikitext-103,"
            echo "                             openwebtext, c4, tinystories, pile-subset,"
            echo "                             fineweb, fineweb-edu, fineweb-edu-10bt"
            echo "  --max-samples N      最大样本数，留空使用推荐值 (默认: 使用推荐值)"
            echo "  --bp-dataset NAME    BP数据集名称 (Calibrate/Instruct模式用)"
            echo "                       留空使用主数据集 (默认: 使用主数据集)"
            echo "  --bp-max-samples N   BP数据集最大样本数 (默认: 使用推荐值)"
            echo "  --instruct-cosine-targets '0.9,0.95'   Instruct模式的余弦目标 (默认: 0.9)"
            echo "  --instruct-noise-scales '0.5,1.0'      Instruct模式的噪声强度 (默认: 0.5)"
            echo "  --use-lr-scheduler   启用余弦退火学习率调度器 (默认: 启用)"
            echo "  --no-lr-scheduler    禁用学习率调度器"
            echo "  --warmup-steps N     学习率预热步数 (默认: 300)"
            echo "  --min-lr VALUE       最小学习率 (默认: 1e-6)"
            echo "  --gradient-accumulation-steps N  梯度累积步数 (仅FO模式, 默认: 1)"
            echo "                       有效batch size = batch_size * gradient_accumulation_steps"
            echo "  -h, --help           显示帮助信息"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 自动检测GPU
if [ -z "$GPU_IDS" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ $GPU_COUNT -gt 0 ]; then
            GPU_IDS=$(seq -s, 0 $((GPU_COUNT-1)))
            echo -e "${BLUE}🔍 Auto-detected $GPU_COUNT GPU(s): $GPU_IDS${NC}"
        else
            echo -e "${YELLOW}⚠️  No GPUs detected, using CPU${NC}"
            GPU_IDS="cpu"
        fi
    else
        echo -e "${YELLOW}⚠️  nvidia-smi not found, using CPU${NC}"
        GPU_IDS="cpu"
    fi
fi

# 创建日志与结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DESCRIPTOR="${MODES}_${SCOPES}_${BATCH_SIZES}_${QUERY_BUDGETS}_${BP_INTERVALS}_${LEARNING_RATES_ZO}_${OPTIMIZERS}_${EPOCHS}_${LOG_INTERVAL}_${INSTRUCT_COSINE_TARGETS}_${INSTRUCT_NOISE_SCALES}"
mkdir -p "$LOGS_ROOT"
RUN_LOG_ROOT="${LOGS_ROOT}/parallel_sweep_${TIMESTAMP}"
EXPERIMENT_LOG_ROOT="${RUN_LOG_ROOT}/experiments"
RESULTS_DIR="${RUN_LOG_ROOT}/results_${RUN_DESCRIPTOR}"
CSV_DIR="${RUN_LOG_ROOT}/csv_logs_${RUN_DESCRIPTOR}"
CACHE_DIR="cache"
TEMP_DIR="${RUN_LOG_ROOT}/temp"

mkdir -p "$RUN_LOG_ROOT" "$EXPERIMENT_LOG_ROOT" "$RESULTS_DIR" "$CSV_DIR" "$CACHE_DIR" "$TEMP_DIR"

LOG_FILE="${RUN_LOG_ROOT}/parallel_sweep.log"
SUMMARY_FILE="${RUN_LOG_ROOT}/parallel_sweep_summary.txt"
JOB_LOG_DIR="${RUN_LOG_ROOT}/job_logs"
PID_FILE="${RUN_LOG_ROOT}/parallel_sweep.pids"
STATUS_FILE="${RUN_LOG_ROOT}/parallel_sweep.status"
mkdir -p "$JOB_LOG_DIR"

# 清理函数
cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true
    
    echo ""
    echo -e "${YELLOW}⚠️  收到退出信号，正在清理所有子进程...${NC}"
    echo "清理时间: $(date)" >> "$LOG_FILE"
    
    # 从 PID 文件读取所有子进程
    if [ -f "$PID_FILE" ]; then
        echo "从 PID 文件读取进程列表: $PID_FILE" >> "$LOG_FILE"
        while IFS= read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "终止进程: $pid" >> "$LOG_FILE"
                kill -TERM "$pid" 2>/dev/null || true
                ALL_CHILD_PIDS+=("$pid")
            fi
        done < "$PID_FILE"
    fi
    
    # 等待所有进程退出
    local wait_count=0
    for pid in "${ALL_CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "等待进程 $pid 退出..." | tee -a "$LOG_FILE"
            wait_count=$((wait_count + 1))
        fi
    done
    
    if [ $wait_count -gt 0 ]; then
        echo "等待 $wait_count 个进程退出 (最多10秒)..." | tee -a "$LOG_FILE"
        sleep 2
        
        # 强制终止仍在运行的进程
        for pid in "${ALL_CHILD_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "强制终止进程: $pid" | tee -a "$LOG_FILE"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        sleep 1
    fi
    
    echo -e "${GREEN}✅ 清理完成${NC}"
    echo "清理完成时间: $(date)" >> "$LOG_FILE"
    
    # 更新状态文件
    if [ -f "$STATUS_FILE" ]; then
        echo "STOPPED_AT=$(date)" >> "$STATUS_FILE"
    fi
}

# 注册信号处理
trap cleanup EXIT INT TERM QUIT

# 初始化状态文件
echo "MAIN_PID=$MAIN_PID" > "$STATUS_FILE"
echo "STARTED_AT=$(date)" >> "$STATUS_FILE"
echo "PID_FILE=$PID_FILE" >> "$STATUS_FILE"
echo "LOG_FILE=$LOG_FILE" >> "$STATUS_FILE"

echo -e "${BLUE}🚀 Starting Parallel ZO vs FO Parameter Sweep${NC}"
echo -e "${BLUE}============================================${NC}"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "GPU IDs: $GPU_IDS"
echo "Model sizes: ${MODEL_SIZES[*]}"
echo "Dataset: $DATASET"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples: $MAX_SAMPLES"
else
    echo "Max samples: Using recommended value"
fi
if [ -n "$BP_DATASET" ]; then
    echo "BP Dataset: $BP_DATASET"
    if [ -n "$BP_MAX_SAMPLES" ]; then
        echo "BP Max samples: $BP_MAX_SAMPLES"
    else
        echo "BP Max samples: Using recommended value"
    fi
else
    echo "BP Dataset: Same as main dataset"
fi
echo "Instruct cosine targets: ${INSTRUCT_COSINE_TARGETS[*]}"
echo "Instruct noise scales: ${INSTRUCT_NOISE_SCALES[*]}"
echo "Results will be saved to: $RESULTS_DIR"
echo "CSV logs will be saved to: $CSV_DIR"
echo "Dataset cache: $CACHE_DIR"
echo "Log file: $LOG_FILE"
echo "Run logs directory: $RUN_LOG_ROOT"
echo "Job logs directory: $JOB_LOG_DIR"
echo ""

# 生成所有实验配置
generate_experiments() {
    local experiments=()
    local exp_id=0
    
    for model_size in "${MODEL_SIZES[@]}"; do
        for mode in "${MODES[@]}"; do
            for scope in "${SCOPES[@]}"; do
                for batch_size in "${BATCH_SIZES[@]}"; do
                    for block_size in "${BLOCK_SIZES[@]}"; do
                        for optimizer in "${OPTIMIZERS[@]}"; do
                            if [ "$mode" = "FO" ]; then
                                for lr in "${LEARNING_RATES_ZO[@]}"; do
                                    experiments+=("$exp_id:$mode:$scope:$batch_size:$block_size:N/A:$lr:$optimizer:N/A:$model_size:N/A:N/A")
                                    exp_id=$((exp_id + 1))
                                done
                            elif [ "$mode" = "ZO" ]; then
                                for q in "${QUERY_BUDGETS[@]}"; do
                                    for lr in "${LEARNING_RATES_ZO[@]}"; do
                                        experiments+=("$exp_id:$mode:$scope:$batch_size:$block_size:$q:$lr:$optimizer:N/A:$model_size:N/A:N/A")
                                        exp_id=$((exp_id + 1))
                                    done
                                done
                            elif [ "$mode" = "Instruct" ]; then
                                for q in "${QUERY_BUDGETS[@]}"; do
                                    for lr in "${LEARNING_RATES_ZO[@]}"; do
                                        for bp_interval in "${BP_INTERVALS[@]}"; do
                                            for cos_target in "${INSTRUCT_COSINE_TARGETS[@]}"; do
                                                for noise_scale in "${INSTRUCT_NOISE_SCALES[@]}"; do
                                                    experiments+=("$exp_id:$mode:$scope:$batch_size:$block_size:$q:$lr:$optimizer:$bp_interval:$model_size:$cos_target:$noise_scale")
                                                    exp_id=$((exp_id + 1))
                                                done
                                            done
                                        done
                                    done
                                done
                            else
                                for q in "${QUERY_BUDGETS[@]}"; do
                                    for lr in "${LEARNING_RATES_ZO[@]}"; do
                                        for bp_interval in "${BP_INTERVALS[@]}"; do
                                            experiments+=("$exp_id:$mode:$scope:$batch_size:$block_size:$q:$lr:$optimizer:$bp_interval:$model_size:N/A:N/A")
                                            exp_id=$((exp_id + 1))
                                        done
                                    done
                                done
                            fi
                        done
                    done
                done
            done
        done
    done
    
    printf '%s\n' "${experiments[@]}"
}

# 运行单个实验
run_single_experiment() {
    local exp_config="$1"
    local gpu_id="$2"
    
    IFS=':' read -r exp_id mode scope batch_size block_size q lr optimizer bp_interval model_size cos_target noise_scale <<< "$exp_config"
    
    # 将 N/A 替换为 NA 以避免文件路径问题
    local q_safe="${q//\//_}"
    local bp_safe="${bp_interval//\//_}"
    local cos_safe="${cos_target//\//_}"
    local noise_safe="${noise_scale//\//_}"
    local cos_label=""
    local noise_label=""
    if [ "$cos_target" != "N/A" ]; then
        cos_label="_ct${cos_safe}"
    fi
    if [ "$noise_scale" != "N/A" ]; then
        noise_label="_ns${noise_safe}"
    fi
    local exp_name="${mode}_${model_size}_${scope}_bs${batch_size}_blk${block_size}_q${q_safe}_bp${bp_safe}_opt${optimizer}_lr${lr}${cos_label}${noise_label}"
    local csv_file="${CSV_DIR}/${exp_name}.csv"
    local job_log="${JOB_LOG_DIR}/${exp_name}.log"
    local exp_log_dir="${EXPERIMENT_LOG_ROOT}/${exp_name}"
    local checkpoint_dir="${exp_log_dir}/checkpoint"
    local run_pid="${BASHPID:-$$}"
    
    mkdir -p "$exp_log_dir"
    echo -e "${YELLOW}📊 Starting experiment: $exp_name (GPU: $gpu_id, PID: $run_pid)${NC}" | tee -a "$job_log"
    
    # 构建命令
    local cmd="python reproduce_zo_paper.py"
    cmd="$cmd --mode $mode"
    cmd="$cmd --scope $scope"
    cmd="$cmd --batch_size $batch_size"
    cmd="$cmd --block_size $block_size"
    cmd="$cmd --learning_rate $lr"
    cmd="$cmd --optimizer $optimizer"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --csv_file $csv_file"
    cmd="$cmd --log_interval $LOG_INTERVAL"
    cmd="$cmd --run_name $exp_name"
    cmd="$cmd --log_dir $exp_log_dir"
    cmd="$cmd --checkpoint_dir $checkpoint_dir"
    
    # 注意: 模型和数据集配置目前在Python脚本中硬编码
    # 如需使用不同配置，请直接修改 reproduce_zo_paper.py 中的配置
    cmd="$cmd --model_size $model_size"
    cmd="$cmd --dataset $DATASET"
    if [ -n "$MAX_SAMPLES" ]; then
        cmd="$cmd --max_samples $MAX_SAMPLES"
    fi
    
    # BP数据集配置（用于Calibrate/Instruct模式）
    if [ -n "$BP_DATASET" ]; then
        cmd="$cmd --bp_dataset $BP_DATASET"
    fi
    if [ -n "$BP_MAX_SAMPLES" ]; then
        cmd="$cmd --bp_max_samples $BP_MAX_SAMPLES"
    fi
    
    if [[ "$mode" == "ZO" || "$mode" == "Calibrate" || "$mode" == "Instruct" ]] && [ "$q" != "N/A" ]; then
        cmd="$cmd --query_budget_q $q"
    fi
    if [ "$bp_interval" != "N/A" ]; then
        cmd="$cmd --bp_interval $bp_interval"
    fi
    if [ "$mode" = "Instruct" ] && [ "$cos_target" != "N/A" ]; then
        cmd="$cmd --instruct_cosine_target $cos_target"
    fi
    if [ "$mode" = "Instruct" ] && [ "$noise_scale" != "N/A" ]; then
        cmd="$cmd --instruct_noise_scale $noise_scale"
    fi
    
    # 学习率调度器参数
    if [ "$USE_LR_SCHEDULER" = true ]; then
        cmd="$cmd --use_lr_scheduler"
        cmd="$cmd --warmup_steps $WARMUP_STEPS"
        cmd="$cmd --min_lr $MIN_LR"
    fi
    
    # 梯度累积参数（仅FO模式）
    if [ "$mode" = "FO" ] && [ "$GRADIENT_ACCUMULATION_STEPS" -gt 1 ]; then
        cmd="$cmd --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
    fi
    
    # 设置GPU环境变量
    if [ "$gpu_id" != "cpu" ]; then
        export CUDA_VISIBLE_DEVICES="$gpu_id"
    else
        unset CUDA_VISIBLE_DEVICES
    fi
    
    echo "Command: $cmd" >> "$job_log"
    echo "GPU: $gpu_id" >> "$job_log"
    echo "Shell PID: $run_pid" >> "$job_log"
    echo "CSV file: $csv_file" >> "$job_log"
    echo "Experiment log dir: $exp_log_dir" >> "$job_log"
    echo "Checkpoint dir: $checkpoint_dir" >> "$job_log"
    if [ "$mode" = "Instruct" ]; then
        echo "Instruct cosine target: $cos_target" >> "$job_log"
        echo "Instruct noise scale: $noise_scale" >> "$job_log"
    fi
    echo "Start time: $(date)" >> "$job_log"
    echo "----------------------------------------" >> "$job_log"
    
    # 运行实验
    eval $cmd >> "$job_log" 2>&1 &
    local child_pid=$!
    echo "Command PID: $child_pid" | tee -a "$job_log"
    
    # 记录 PID 到文件和全局数组
    echo "$child_pid" >> "$PID_FILE"
    ALL_CHILD_PIDS+=("$child_pid")
    
    # 记录到状态文件
    echo "PID_${child_pid}=${exp_name}" >> "$STATUS_FILE"

    # 等待子进程完成（不管成功还是失败都继续）
    wait $child_pid 2>/dev/null
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Experiment $exp_name completed successfully${NC}" | tee -a "$job_log"
        echo "End time: $(date)" >> "$job_log"
        echo "SUCCESS" >> "$job_log"
    else
        # 子进程失败不影响其他进程，只记录失败信息
        echo -e "${RED}❌ Experiment $exp_name failed with exit code $exit_code${NC}" | tee -a "$job_log"
        echo "End time: $(date)" >> "$job_log"
        echo "FAILED (exit code: $exit_code)" >> "$job_log"
        # 注意：这里不要 exit 或 return 非零值，让其他实验继续运行
    fi
    
    # 从状态文件中移除
    sed -i "/^PID_${child_pid}=/d" "$STATUS_FILE" 2>/dev/null || true

    # 总是返回 0，单个实验失败不影响整体流程
    return 0
}

# 并行执行实验
run_parallel_experiments() {
    local experiments=($(generate_experiments))
    local total_experiments=${#experiments[@]}
    local completed=0
    local successful=0
    local failed=0
    
    echo -e "${BLUE}📋 Generated $total_experiments experiments${NC}"
    echo ""
    
    # 将GPU ID转换为数组（支持逗号和空格分隔）
    if [[ "$GPU_IDS" == *","* ]]; then
        IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    else
        IFS=' ' read -ra GPU_ARRAY <<< "$GPU_IDS"
    fi
    local gpu_count=${#GPU_ARRAY[@]}
    local gpu_index=0
    
    # 创建任务队列
    local job_queue=()
    local running_jobs=()
    
    # 初始化任务队列
    for exp in "${experiments[@]}"; do
        job_queue+=("$exp")
    done
    
    echo -e "${BLUE}🚀 Starting parallel execution...${NC}"
    echo ""
    
    # 主循环：管理并行任务
    while [ $completed -lt $total_experiments ]; do
        # 启动新任务（如果队列不为空且未达到最大并行数）
        while [ ${#running_jobs[@]} -lt $MAX_PARALLEL_JOBS ] && [ ${#job_queue[@]} -gt 0 ]; do
            local exp="${job_queue[0]}"
            job_queue=("${job_queue[@]:1}")  # 移除第一个元素
            
            local gpu_id="${GPU_ARRAY[$gpu_index]}"
            gpu_index=$(((gpu_index + 1) % gpu_count))
            
            # 在后台运行实验
            run_single_experiment "$exp" "$gpu_id" &
            local pid=$!
            running_jobs+=("$pid:$exp:$gpu_id")
            
            echo -e "${PURPLE}🔄 Started job $pid for experiment $exp on GPU $gpu_id${NC}"
        done
        
        # 检查完成的任务
        local new_running_jobs=()
        for job in "${running_jobs[@]}"; do
            IFS=':' read -r pid exp gpu_id <<< "$job"
            if kill -0 $pid 2>/dev/null; then
                # 任务仍在运行
                new_running_jobs+=("$job")
            else
                # 任务已完成（不管成功还是失败）
                local exit_code=0
                # 使用 wait 获取退出码，即使失败也不中断
                wait $pid 2>/dev/null || exit_code=$?
                
                completed=$((completed + 1))
                
                if [ $exit_code -eq 0 ]; then
                    successful=$((successful + 1))
                else
                    # 单个任务失败不影响其他任务
                    failed=$((failed + 1))
                fi
                
                echo -e "${BLUE}📊 Progress: $completed/$total_experiments completed (Success: $successful, Failed: $failed)${NC}"
            fi
        done
        running_jobs=("${new_running_jobs[@]}")
        
        # 更新状态文件
        echo "PROGRESS=$completed/$total_experiments" > "${STATUS_FILE}.tmp"
        echo "SUCCESS=$successful" >> "${STATUS_FILE}.tmp"
        echo "FAILED=$failed" >> "${STATUS_FILE}.tmp"
        echo "RUNNING=${#running_jobs[@]}" >> "${STATUS_FILE}.tmp"
        cat "$STATUS_FILE" >> "${STATUS_FILE}.tmp"
        mv "${STATUS_FILE}.tmp" "$STATUS_FILE"
        
        # 短暂等待
        sleep 2
    done
    
    # 等待所有剩余任务完成
    if [ ${#running_jobs[@]} -gt 0 ]; then
        echo -e "${YELLOW}等待 ${#running_jobs[@]} 个剩余任务完成...${NC}"
        for job in "${running_jobs[@]}"; do
            IFS=':' read -r pid exp gpu_id <<< "$job"
            local exit_code=0
            # 等待每个进程，即使失败也继续处理其他进程
            wait $pid 2>/dev/null || exit_code=$?
            
            completed=$((completed + 1))
            
            if [ $exit_code -eq 0 ]; then
                successful=$((successful + 1))
            else
                # 单个任务失败不影响其他任务
                failed=$((failed + 1))
            fi
        done
    fi
    
    # 导出结果供主函数使用
    echo "$successful" > "$TEMP_DIR/successful_count"
    echo "$failed" > "$TEMP_DIR/failed_count"
    echo "$total_experiments" > "$TEMP_DIR/total_count"
    
    echo ""
    echo -e "${GREEN}🎉 All experiments completed!${NC}"
    echo "Total: $total_experiments, Success: $successful, Failed: $failed"
}

# 生成最终报告
generate_final_report() {
    local start_time=$1
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    # 从文件读取结果
    local successful=0
    local failed=0
    local total_experiments=0
    
    if [ -f "$TEMP_DIR/successful_count" ]; then
        successful=$(cat "$TEMP_DIR/successful_count")
    fi
    if [ -f "$TEMP_DIR/failed_count" ]; then
        failed=$(cat "$TEMP_DIR/failed_count")
    fi
    if [ -f "$TEMP_DIR/total_count" ]; then
        total_experiments=$(cat "$TEMP_DIR/total_count")
    fi
    
    local success_rate=0
    if [ $total_experiments -gt 0 ]; then
        success_rate=$(( successful * 100 / total_experiments ))
    fi
    
    echo -e "${BLUE}📋 PARALLEL SWEEP SUMMARY REPORT${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
    echo "GPU IDs used: $GPU_IDS"
    echo "Model sizes: ${MODEL_SIZES[*]}"
    echo "Dataset: $DATASET"
    if [ -n "$BP_DATASET" ]; then
        echo "BP Dataset: $BP_DATASET"
    fi
    echo "Instruct cosine targets: ${INSTRUCT_COSINE_TARGETS[*]}"
    echo "Instruct noise scales: ${INSTRUCT_NOISE_SCALES[*]}"
    echo "Total experiments: $total_experiments"
    echo -e "Successful: ${GREEN}$successful${NC}"
    echo -e "Failed: ${RED}$failed${NC}"
    echo "Success rate: ${success_rate}%"
    echo "Total time: ${hours}h ${minutes}m ${seconds}s"
    echo ""
    echo "Results directory: $RESULTS_DIR"
    echo "CSV logs directory: $CSV_DIR"
    echo "Job logs directory: $JOB_LOG_DIR"
    echo "Log file: $LOG_FILE"
    echo "PID file: $PID_FILE"
    echo "Status file: $STATUS_FILE"
    echo "Summary file: $SUMMARY_FILE"
    echo ""
    
    # 保存到摘要文件
    {
        echo "PARALLEL SWEEP SUMMARY REPORT"
        echo "================================="
        echo "Timestamp: $(date)"
        echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
        echo "GPU IDs used: $GPU_IDS"
        echo "Model sizes: ${MODEL_SIZES[*]}"
        echo "Dataset: $DATASET"
        if [ -n "$BP_DATASET" ]; then
            echo "BP Dataset: $BP_DATASET"
        fi
        echo "Instruct cosine targets: ${INSTRUCT_COSINE_TARGETS[*]}"
        echo "Instruct noise scales: ${INSTRUCT_NOISE_SCALES[*]}"
        echo "Total experiments: $total_experiments"
        echo "Successful: $successful"
        echo "Failed: $failed"
        echo "Success rate: ${success_rate}%"
        echo "Total time: ${hours}h ${minutes}m ${seconds}s"
        echo ""
        echo "Results directory: $RESULTS_DIR"
        echo "CSV logs directory: $CSV_DIR"
        echo "Job logs directory: $JOB_LOG_DIR"
        echo "Log file: $LOG_FILE"
        echo "PID file: $PID_FILE"
        echo "Status file: $STATUS_FILE"
    } > "$SUMMARY_FILE"
    
    # 列出所有结果文件
    echo -e "${BLUE}📁 Generated Files:${NC}"
    echo "PNG plots:"
    ls -la "$RESULTS_DIR"/*.png 2>/dev/null | head -10 || echo "  No PNG files found"
    if [ $(ls -1 "$RESULTS_DIR"/*.png 2>/dev/null | wc -l) -gt 10 ]; then
        echo "  ... and $(($(ls -1 "$RESULTS_DIR"/*.png 2>/dev/null | wc -l) - 10)) more files"
    fi
    echo ""
    echo "CSV logs:"
    ls -la "$CSV_DIR"/*.csv 2>/dev/null | head -10 || echo "  No CSV files found"
    if [ $(ls -1 "$CSV_DIR"/*.csv 2>/dev/null | wc -l) -gt 10 ]; then
        echo "  ... and $(($(ls -1 "$CSV_DIR"/*.csv 2>/dev/null | wc -l) - 10)) more files"
    fi
    echo ""
}

# 主程序
main() {
    local start_time=$(date +%s)
    
    # 记录配置
    echo "Configuration:" >> "$LOG_FILE"
    echo "MODES: ${MODES[*]}" >> "$LOG_FILE"
    echo "SCOPES: ${SCOPES[*]}" >> "$LOG_FILE"
    echo "BATCH_SIZES: ${BATCH_SIZES[*]}" >> "$LOG_FILE"
    echo "QUERY_BUDGETS: ${QUERY_BUDGETS[*]}" >> "$LOG_FILE"
    echo "BP_INTERVALS: ${BP_INTERVALS[*]}" >> "$LOG_FILE"
    echo "LEARNING_RATES_ZO: ${LEARNING_RATES_ZO[*]}" >> "$LOG_FILE"
    echo "OPTIMIZERS: ${OPTIMIZERS[*]}" >> "$LOG_FILE"
    echo "EPOCHS: $EPOCHS" >> "$LOG_FILE"
    echo "MODEL_SIZES: ${MODEL_SIZES[*]}" >> "$LOG_FILE"
    echo "DATASET: $DATASET" >> "$LOG_FILE"
    echo "MAX_SAMPLES: ${MAX_SAMPLES:-auto}" >> "$LOG_FILE"
    echo "BP_DATASET: ${BP_DATASET:-same_as_main}" >> "$LOG_FILE"
    echo "BP_MAX_SAMPLES: ${BP_MAX_SAMPLES:-auto}" >> "$LOG_FILE"
    echo "INSTRUCT_COSINE_TARGETS: ${INSTRUCT_COSINE_TARGETS[*]}" >> "$LOG_FILE"
    echo "INSTRUCT_NOISE_SCALES: ${INSTRUCT_NOISE_SCALES[*]}" >> "$LOG_FILE"
    echo "MAX_PARALLEL_JOBS: $MAX_PARALLEL_JOBS" >> "$LOG_FILE"
    echo "GPU_IDS: $GPU_IDS" >> "$LOG_FILE"
    echo "=========================================" >> "$LOG_FILE"
    
    # 运行并行实验（不要因为单个实验失败而中断）
    run_parallel_experiments 2>&1 | tee -a "$LOG_FILE"
    local run_exit_code=${PIPESTATUS[0]}
    
    # 生成报告
    generate_final_report "$start_time" 2>&1 | tee -a "$LOG_FILE"
    
    # 检查是否有失败的实验
    local failed_count=0
    if [ -f "$TEMP_DIR/failed_count" ]; then
        failed_count=$(cat "$TEMP_DIR/failed_count")
    fi
    
    if [ $failed_count -eq 0 ]; then
        echo -e "${GREEN}🎉 Parallel sweep completed successfully! All experiments passed.${NC}"
    else
        echo -e "${YELLOW}⚠️  Parallel sweep completed. $failed_count experiment(s) failed.${NC}"
        echo -e "${YELLOW}    Check individual job logs in $JOB_LOG_DIR for details.${NC}"
    fi
    echo "Check the results in the $RESULTS_DIR and $CSV_DIR directories."
    echo "Detailed logs available in: $LOG_FILE"
    echo "PID tracking file: $PID_FILE"
    echo "Status file: $STATUS_FILE"
    
    # 即使有失败的实验，也返回 0（整体流程成功完成）
    # 如果需要根据失败数量返回非零，可以取消下面的注释
    # [ $failed_count -eq 0 ] && return 0 || return 1
    return 0
}

# 运行主程序
main "$@"
