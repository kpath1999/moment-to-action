#!/bin/bash

# Configuration
MODEL="NexaAI/Qwen3-0.6B-GGUF"
OUTPUT_FILE="inference_results_$(date +%Y%m%d_%H%M%S).jsonl"
SUMMARY_FILE="benchmark_summary_$(date +%Y%m%d_%H%M%S).txt"
API_PID=""

# Test prompts - each will be tested once
PROMPTS=(
    "What is the capital of France?"
    "Explain quantum computing in simple terms."
    "Write a Python function to calculate fibonacci numbers."
)

# Function to get memory usage in MB
get_memory_usage() {
    if [ -n "$API_PID" ]; then
        ps -p $API_PID -o rss= 2>/dev/null | awk '{print $1/1024}'
    else
        echo "0"
    fi
}

# Function to get CPU usage percentage
get_cpu_usage() {
    if [ -n "$API_PID" ]; then
        ps -p $API_PID -o %cpu= 2>/dev/null | awk '{print $1}'
    else
        echo "0"
    fi
}

# Function to get power consumption (Qualcomm-specific)
get_power_usage() {
    # Try multiple power sources
    if [ -f /sys/class/power_supply/battery/power_now ]; then
        cat /sys/class/power_supply/battery/power_now 2>/dev/null | awk '{print $1/1000000}'
    elif [ -f /sys/class/power_supply/BAT0/power_now ]; then
        cat /sys/class/power_supply/BAT0/power_now 2>/dev/null | awk '{print $1/1000000}'
    else
        echo "N/A"
    fi
}

# Function to check GPU usage (Qualcomm Adreno)
get_gpu_usage() {
    # Adreno GPU usage
    if [ -f /sys/class/kgsl/kgsl-3d0/gpubusy ]; then
        cat /sys/class/kgsl/kgsl-3d0/gpubusy 2>/dev/null | awk '{print ($1/$2)*100}'
    elif [ -f /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage ]; then
        cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage 2>/dev/null
    else
        echo "N/A"
    fi
}

# Function to get GPU frequency
get_gpu_freq() {
    if [ -f /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq ]; then
        FREQ=$(cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq 2>/dev/null)
        echo "scale=0; $FREQ/1000000" | bc 2>/dev/null || echo "N/A"
    else
        echo "N/A"
    fi
}

# Function to check NPU/DSP usage (Qualcomm Hexagon)
get_npu_usage() {
    # Check for NPU/CDSP activity
    if [ -d /sys/kernel/debug/msm_npu ]; then
        # NPU debug interface exists
        echo "Active"
    elif [ -f /sys/class/misc/msm_npu/device/power/runtime_active_time ]; then
        cat /sys/class/misc/msm_npu/device/power/runtime_active_time 2>/dev/null || echo "N/A"
    else
        echo "N/A"
    fi
}

# Function to get thermal readings for different processors
get_thermal_readings() {
    local PREFIX=$1
    echo "--- Thermal Zones ($PREFIX) ---" >> "$SUMMARY_FILE"
    
    for zone in /sys/class/thermal/thermal_zone*/type; do
        if [ -f "$zone" ]; then
            ZONE_TYPE=$(cat "$zone")
            ZONE_DIR=$(dirname "$zone")
            TEMP=$(cat "$ZONE_DIR/temp" 2>/dev/null)
            if [ -n "$TEMP" ]; then
                TEMP_C=$(echo "scale=1; $TEMP/1000" | bc)
                echo "$ZONE_TYPE: ${TEMP_C}°C" >> "$SUMMARY_FILE"
            fi
        fi
    done
    echo "" >> "$SUMMARY_FILE"
}

# Function to detect active accelerator
detect_accelerator() {
    local gpu_usage=$(get_gpu_usage)
    local npu_usage=$(get_npu_usage)
    
    # Heuristic: if GPU usage > 10%, likely using GPU
    if [ "$gpu_usage" != "N/A" ] && [ "$(echo "$gpu_usage > 10" | bc 2>/dev/null)" = "1" ]; then
        echo "GPU"
    elif [ "$npu_usage" = "Active" ]; then
        echo "NPU"
    else
        echo "CPU"
    fi
}

# Get processor information
get_processor_info() {
    echo "=== Processor Information ===" >> "$SUMMARY_FILE"
    cat /proc/cpuinfo | grep -E "processor|model name|Hardware|cpu MHz" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # GPU Information
    echo "=== GPU Information ===" >> "$SUMMARY_FILE"
    if [ -d /sys/class/kgsl/kgsl-3d0 ]; then
        echo "GPU Device: Qualcomm Adreno" >> "$SUMMARY_FILE"
        if [ -f /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq ]; then
            echo "Current Freq: $(get_gpu_freq) MHz" >> "$SUMMARY_FILE"
        fi
        if [ -f /sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies ]; then
            echo "Available Freqs: $(cat /sys/class/kgsl/kgsl-3d0/devfreq/available_frequencies 2>/dev/null | awk '{for(i=1;i<=NF;i++) printf "%d ", $i/1000000}')" >> "$SUMMARY_FILE"
        fi
    else
        echo "GPU: Not detected" >> "$SUMMARY_FILE"
    fi
    echo "" >> "$SUMMARY_FILE"
    
    # NPU/DSP Information
    echo "=== NPU/DSP Information ===" >> "$SUMMARY_FILE"
    if [ -d /sys/kernel/debug/msm_npu ] || [ -d /sys/class/misc/msm_npu ]; then
        echo "NPU Device: Qualcomm Hexagon DSP/NPU" >> "$SUMMARY_FILE"
    else
        echo "NPU: Not detected or requires root access" >> "$SUMMARY_FILE"
    fi
    echo "" >> "$SUMMARY_FILE"
    
    # Initial thermal readings
    get_thermal_readings "Initial"
}

# Find the API server PID
find_api_pid() {
    # Look for nexa or python process running the model server
    API_PID=$(ps aux | grep -E "(nexa|python.*18181)" | grep -v grep | awk '{print $2}' | head -1)
    if [ -z "$API_PID" ]; then
        echo "Warning: Could not find API server process"
    else
        echo "Found API server PID: $API_PID"
    fi
}

echo "Starting benchmark: Testing ${#PROMPTS[@]} prompts"
echo "Results will be saved to: $OUTPUT_FILE"
echo "Summary will be saved to: $SUMMARY_FILE"
echo ""

# Find API server process
find_api_pid

# Get initial system information
get_processor_info

# Header for summary
echo "=== Benchmark Summary ===" > "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Model: $MODEL" >> "$SUMMARY_FILE"
echo "Number of prompts: ${#PROMPTS[@]}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Test each prompt once
for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    ITERATION=$((i + 1))
    
    echo "----------------------------------------"
    echo "Test $ITERATION/${#PROMPTS[@]}"
    echo "Prompt: $PROMPT"
    echo "----------------------------------------"
    
    # Get baseline metrics
    MEM_BEFORE=$(get_memory_usage)
    CPU_BEFORE=$(get_cpu_usage)
    POWER_BEFORE=$(get_power_usage)
    GPU_BEFORE=$(get_gpu_usage)
    GPU_FREQ_BEFORE=$(get_gpu_freq)
    
    # Record start time
    START_TIME=$(date +%s.%N)
    
    # Run inference via API
    RESPONSE=$(curl -s http://127.0.0.1:18181/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"max_tokens\": 100,
            \"temperature\": 0.7
        }")
    
    # Record end time
    END_TIME=$(date +%s.%N)
    LATENCY=$(echo "$END_TIME - $START_TIME" | bc)
    
    # Get post-inference metrics
    MEM_AFTER=$(get_memory_usage)
    CPU_AFTER=$(get_cpu_usage)
    POWER_AFTER=$(get_power_usage)
    GPU_AFTER=$(get_gpu_usage)
    GPU_FREQ_AFTER=$(get_gpu_freq)
    
    # Detect which accelerator was used
    ACCELERATOR=$(detect_accelerator)
    
    # Calculate average memory and CPU during inference
    MEM_AVG=$(echo "($MEM_BEFORE + $MEM_AFTER) / 2" | bc -l)
    CPU_AVG=$(echo "($CPU_BEFORE + $CPU_AFTER) / 2" | bc -l)
    GPU_AVG=$(echo "($GPU_BEFORE + $GPU_AFTER) / 2" | bc -l 2>/dev/null || echo "N/A")
    GPU_FREQ_AVG=$(echo "($GPU_FREQ_BEFORE + $GPU_FREQ_AFTER) / 2" | bc -l 2>/dev/null || echo "N/A")
    
    # Extract token counts from response
    COMPLETION_TOKENS=$(echo "$RESPONSE" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*')
    PROMPT_TOKENS=$(echo "$RESPONSE" | grep -o '"prompt_tokens":[0-9]*' | grep -o '[0-9]*')
    TOTAL_TOKENS=$(echo "$RESPONSE" | grep -o '"total_tokens":[0-9]*' | grep -o '[0-9]*')
    
    # Calculate tokens per second
    if [ -n "$COMPLETION_TOKENS" ] && [ "$COMPLETION_TOKENS" != "0" ]; then
        TOKENS_PER_SEC=$(echo "scale=2; $COMPLETION_TOKENS / $LATENCY" | bc)
    else
        TOKENS_PER_SEC="N/A"
    fi
    
    # Save detailed results to JSONL
    echo "{\"iteration\": $ITERATION, \"prompt\": \"$PROMPT\", \"latency\": $LATENCY, \"accelerator\": \"$ACCELERATOR\", \"memory_mb\": $MEM_AVG, \"cpu_percent\": $CPU_AVG, \"gpu_percent\": \"$GPU_AVG\", \"gpu_freq_mhz\": \"$GPU_FREQ_AVG\", \"power_watts\": \"$POWER_AFTER\", \"prompt_tokens\": $PROMPT_TOKENS, \"completion_tokens\": $COMPLETION_TOKENS, \"total_tokens\": $TOTAL_TOKENS, \"tokens_per_sec\": \"$TOKENS_PER_SEC\", \"response\": $RESPONSE}" >> "$OUTPUT_FILE"
    
    # Print metrics
    echo "  Accelerator: $ACCELERATOR"
    echo "  Latency: ${LATENCY}s"
    echo "  Memory Usage: ${MEM_AVG} MB"
    echo "  CPU Usage: ${CPU_AVG}%"
    echo "  GPU Usage: ${GPU_AVG}% @ ${GPU_FREQ_AVG} MHz"
    echo "  Power: ${POWER_AFTER} W"
    echo "  Tokens: Prompt=$PROMPT_TOKENS, Completion=$COMPLETION_TOKENS, Total=$TOTAL_TOKENS"
    echo "  Throughput: $TOKENS_PER_SEC tokens/sec"
    echo ""
    
    # Add to summary
    echo "--- Test $ITERATION ---" >> "$SUMMARY_FILE"
    echo "Prompt: $PROMPT" >> "$SUMMARY_FILE"
    echo "Accelerator Used: $ACCELERATOR" >> "$SUMMARY_FILE"
    echo "Latency: ${LATENCY}s" >> "$SUMMARY_FILE"
    echo "Memory Usage: ${MEM_AVG} MB" >> "$SUMMARY_FILE"
    echo "CPU Usage: ${CPU_AVG}%" >> "$SUMMARY_FILE"
    echo "GPU Usage: ${GPU_AVG}% @ ${GPU_FREQ_AVG} MHz" >> "$SUMMARY_FILE"
    echo "Power Consumption: ${POWER_AFTER} W" >> "$SUMMARY_FILE"
    echo "Tokens: Prompt=$PROMPT_TOKENS, Completion=$COMPLETION_TOKENS" >> "$SUMMARY_FILE"
    echo "Throughput: $TOKENS_PER_SEC tokens/sec" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    sleep 2  # Delay between requests to allow system to stabilize
done

# Generate comparison summary
echo "=== Comparison Summary ===" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Analyzing results..." >> "$SUMMARY_FILE"

# Final thermal readings
get_thermal_readings "Final"

# Parse JSONL and create comparison
awk 'BEGIN {print "\nPrompt\t\t\t\t\tAccel\tLatency(s)\tTokens/sec\tCPU(%)\tGPU(%)"}
{
    match($0, /"prompt": "([^"]*)"/, prompt);
    match($0, /"accelerator": "([^"]*)"/, accel);
    match($0, /"latency": ([0-9.]+)/, latency);
    match($0, /"tokens_per_sec": "([^"]*)"/, tps);
    match($0, /"cpu_percent": ([0-9.]+)/, cpu);
    match($0, /"gpu_percent": "([^"]*)"/, gpu);
    printf "%-40s\t%s\t%.2f\t\t%s\t\t%.1f\t%s\n", substr(prompt[1], 1, 40), accel[1], latency[1], tps[1], cpu[1], gpu[1]
}' "$OUTPUT_FILE" >> "$SUMMARY_FILE"

echo ""
echo "========================================="
echo "Benchmark complete!"
echo "========================================="
echo "Detailed results: $OUTPUT_FILE"
echo "Summary report: $SUMMARY_FILE"
echo ""
cat "$SUMMARY_FILE"
