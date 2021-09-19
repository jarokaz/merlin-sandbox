FROM nvcr.io/nvidia/merlin/merlin-training:0.6

# Install sshd
RUN apt update && apt install -y openssh-server

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
    
# Create SSH key.
RUN mkdir -p /root/.ssh/ && \
    mkdir -p /var/run/sshd && \
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys


WORKDIR /src

COPY mpi-test.py .

EXPOSE 2222