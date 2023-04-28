ENV TF_NUM_INTEROP_THREADS=1

ENV MKL_ROOT=/opt/intel/oneapi/mkl/latest

ENV LD_PRELOAD=$MKL_ROOT/lib/intel64/libmkl_rt.so

ENV dpcpp_root=${compiler_path}

ENV PATH=$dpcpp_root/bin:${PATH}

ENV LD_LIBRARY_PATH=$dpcpp_root/lib:$dpcpp_root/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}

ENV ITEX_ENABLE_ONEDNN_LAYOUT_OPT=1

ENV OverrideSystolicPipelineSelect=1

