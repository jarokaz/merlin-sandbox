FROM gcr.io/deeplearning-platform-release/base-cu110

ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/lib/x86_64-linux-gnu
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python

RUN conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular cudatoolkit=11.2

ENTRYPOINT ["python"]