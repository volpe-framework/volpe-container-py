FROM python:3.11-slim

# Use a minimal base image and install dependencies needed for Python package builds.
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --prefer-binary -r requirements.txt

COPY ./volpe_py ./volpe_py
COPY . .

CMD ["python", "./main.py"]

