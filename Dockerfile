# Multi-stage build for PageRank with MPI support
FROM ubuntu:22.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    mpich \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source files
COPY code/ ./code/

# Compile serial version
RUN cd code && g++ -O3 -std=c++11 serial_pagerank.cpp -o serial

# Compile MPI version
RUN cd code && mpic++ -O3 -fopenmp -std=c++11 parallel.cpp -o parallel

# Runtime image
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    mpich \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /app/code/serial /app/serial
COPY --from=builder /app/code/parallel /app/parallel

# Copy source for reference
COPY code/ ./code/

# Create data directory
RUN mkdir -p /app/data

# Default command shows usage
CMD ["echo", "Usage: docker run -v /path/to/graph.txt:/app/data/graph.txt pagerank ./serial or mpirun -np 4 ./parallel -f /app/data/graph.txt -n 4 -i 80"]
