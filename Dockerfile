```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc gfortran make git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/lesgourg/class_public.git class
WORKDIR /opt/class
RUN make

WORKDIR /opt
RUN git clone https://github.com/brinckmann/montepython_public.git montepython
WORKDIR /opt/montepython
RUN pip install -r requirements.txt

WORKDIR /workspace
COPY . /workspace/ESTG_MCMC_LISA
RUN pip install -r /workspace/ESTG_MCMC_LISA/requirements.txt

WORKDIR /workspace/ESTG_MCMC_LISA
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
