FROM python:3.10-slim

SHELL ["/bin/bash", "-c"]

RUN pip install --upgrade pip

RUN mkdir /home/f1tenth_benchmarks/
WORKDIR /home/f1tenth_benchmarks/

COPY . /home/f1tenth_benchmarks/
RUN pip install -e .

RUN pip install -r requirements.txt


ENTRYPOINT ["/bin/bash"]
