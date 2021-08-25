FROM intel-optimized-pytorch AS release
COPY --from=intel-optimized-pytorch /root/conda /root/conda
COPY --from=intel-optimized-pytorch /workspace/lib/ /workspace/lib/
COPY --from=intel-optimized-pytorch /root/.local/ /root/.local/
