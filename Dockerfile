from gcr.io/deeplearning-platform-release/base-cu110

RUN git clone https://github.com/NVIDIA/NVTabular.git \
&& conda env create -f=NVTabular/conda/environments/nvtabular_dev_cuda11.2.yml

ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/lib/x86_64-linux-gnu

SHELL ["conda", "run", "-n", "nvtabular_dev_11.2", "/bin/bash", "-c"]

RUN echo $LD_LIBRARY_PATH
RUN python -c "import nvtabular"

