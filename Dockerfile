FROM python:3.10-slim

SHELL ["/bin/bash", "-c"]

RUN pip install --upgrade pip

RUN mkdir /home/f1tenth_sim/
WORKDIR /home/f1tenth_sim/

COPY . /home/f1tenth_sim/
RUN pip install -e .

RUN pip install -r requirements.txt

# does that work bc the simulator is not installed?
ENTRYPOINT ["/bin/bash"]
