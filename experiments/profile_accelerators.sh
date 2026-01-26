#!/bin/bash

# Script to profile Nexa SDK across different accelerators on Rubik Pi 3
# This helps determine which backend (CPU/GPU/NPU) performs best

MODEL="NexaAI/Qwen3-0.6B-GGUF"
TEST_PROMPT="Explain quantum computing in simple terms."
OUTPUT_FILE="accelerator_profile_$(date +%Y%m%d_%H%M%S).txt"

echo "=== Accelerator Performance Profile ===" | tee "$OUTPUT_FILE"
echo "Model: $MODEL" | tee -a "$OUTPUT_FILE"
echo "Test Prompt: $TEST_PROMPT" | tee -a "$OUTPUT_FILE"
echo "Date: $(date)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Function to test inference with specific settings
test_inference() {
    local BACKEND=$1
    local ENV_VARS=$2
    
    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
    echo "Testing: $BACKEND" | tee -a "$OUTPUT_FILE"
    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
    
    START_TIME=$(date +%s.%N)
    
    # Run inference with environment variables
    RESPONSE=$(eval "$ENV_VARS" curl -s http://127.0.0.1:18181/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$TEST_PROMPT\"}],
            \"max_tokens\": 100,
            \"temperature\": 0.7
        }")
    
    END_TIME=$(date +%s.%N)
    LATENCY=$(echo "$END_TIME - $START_TIME" | bc)
    
    # Extract tokens
    COMPLETION_TOKENS=$(echo "$RESPONSE" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*')
    
    if [ -n "$COMPLETION_TOKENS" ] && [ "$COMPLETION_TOKENS" != "0" ]; then
        TOKENS_PER_SEC=$(echo "scale=2; $COMPLETION_TOKENS / $LATENCY" | bc)
    else
        TOKENS_PER_SEC="N/A"
    fi
    
    echo "Latency: ${LATENCY}s" | tee -a "$OUTPUT_FILE"
    echo "Completion Tokens: $COMPLETION_TOKENS" | tee -a "$OUTPUT_FILE"
    echo "Throughput: $TOKENS_PER_SEC tokens/sec" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    sleep 3  # Cool down between tests
}

# Check system capabilities
echo "=== System Capabilities ===" | tee -a "$OUTPUT_FILE"

# CPU info
CPU_MODEL=$(cat /proc/cpuinfo | grep "Hardware" | head -1 | cut -d: -f2 | xargs)
CPU_CORES=$(cat /proc/cpuinfo | grep processor | wc -l)
echo "CPU: $CPU_MODEL ($CPU_CORES cores)" | tee -a "$OUTPUT_FILE"

# GPU check
if [ -d /sys/class/kgsl/kgsl-3d0 ]; then
    echo "GPU: Qualcomm Adreno (detected)" | tee -a "$OUTPUT_FILE"
else
    echo "GPU: Not detected" | tee -a "$OUTPUT_FILE"
fi

# NPU check
if [ -d /sys/kernel/debug/msm_npu ] || [ -d /sys/class/misc/msm_npu ]; then
    echo "NPU: Qualcomm Hexagon DSP/NPU (detected)" | tee -a "$OUTPUT_FILE"
else
    echo "NPU: Not detected or requires root access" | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"

# Note: Nexa SDK may not support direct backend selection via env vars
# This demonstrates the concept - actual implementation depends on SDK capabilities

echo "=== Performance Tests ===" | tee -a "$OUTPUT_FILE"
echo "Note: Backend selection depends on Nexa SDK configuration" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Test 1: Default (let SDK choose)
test_inference "Default (SDK Auto-Select)" ""

# Test 2: CPU-only hint (if supported)
test_inference "CPU Priority" "OMP_NUM_THREADS=$CPU_CORES"

# Test 3: GPU hint (if supported)
test_inference "GPU Priority" "USE_GPU=1"

echo "=========================================" | tee -a "$OUTPUT_FILE"
echo "Profile complete! Results saved to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Instructions for further optimization
cat >> "$OUTPUT_FILE" << 'EOF'

=== How to Determine Active Accelerator ===

1. Monitor GPU Usage:
   watch -n 0.5 cat /sys/class/kgsl/kgsl-3d0/gpubusy

2. Monitor CPU Usage per Core:
   mpstat -P ALL 1

3. Monitor NPU (requires root):
   sudo cat /sys/kernel/debug/msm_npu/perf

4. Check thermal zones during inference:
   watch -n 0.5 'grep -H . /sys/class/thermal/thermal_zone*/temp'

5. Use Qualcomm Snapdragon Profiler:
   - Connect via ADB
   - Monitor real-time CPU/GPU/DSP usage
   - View power consumption per component

=== Nexa SDK Configuration ===

Check your Nexa SDK settings for accelerator selection:
- Model format (GGUF typically uses CPU)
- For GPU/NPU: May need quantized .tflite or .dlc models
- Check: nexa --help for backend options

To force specific backend, you may need to:
1. Use different model formats (.tflite for GPU/NPU)
2. Configure runtime in Nexa SDK settings
3. Check documentation: https://docs.nexaai.com

EOF

cat "$OUTPUT_FILE"
