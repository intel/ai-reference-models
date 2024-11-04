bench=$(python $MODEL_SCRIPT $COMMON_ARGS --inductor --aot-inductor | grep aoti_bench_bin)
echo $bench
$bench