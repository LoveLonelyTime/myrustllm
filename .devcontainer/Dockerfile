FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install tools
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    build-essential \
    pkg-config \
    gcc \
    g++ \
    gdb \
    make \
    cmake \
    llvm \
    clang \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup component add rustfmt clippy rust-analysis

# Check
RUN gcc --version && \
    g++ --version && \
    cmake --version && \
    rustc --version && \
    cargo --version

ENV DEBIAN_FRONTEND=dialog
