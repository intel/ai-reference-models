FROM intel-optimized-pytorch AS release
COPY --from=intel-optimized-pytorch /root/anaconda3 /root/anaconda3
COPY --from=intel-optimized-pytorch /workspace/lib/ /workspace/lib/
COPY --from=intel-optimized-pytorch /root/.local/ /root/.local/
