FROM gcr.io/deeplearning-platform-release/base-cu110

RUN conda install -c nvidia -c rapidsai -c numba -c conda-forge cupy nvtabular cudatoolkit=11.0

RUN pip install Cython nvidia-pyindex tensorflow==2.4.1

ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python

COPY src/ src/

ENTRYPOINT ["python"]