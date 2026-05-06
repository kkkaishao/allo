FROM ubuntu:24.04 AS llvm-builder

ARG LLVM_TAG=llvmorg-22.1.0
ARG INSTALL_PREFIX=/opt/llvm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates git \
      cmake ninja-build clang lld \
      python3 python3-pip file && \
    rm -rf /var/lib/apt/lists/*

# clone the llvm-project
WORKDIR /tmp
RUN git clone https://github.com/llvm/llvm-project.git --depth 1 --branch ${LLVM_TAG} --filter=blob:none

# build and install llvm
RUN cd /tmp/llvm-project && \
    mkdir build && cd build && \
    cmake -G Ninja ../llvm \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DLLVM_ENABLE_PROJECTS="clang;mlir" \
        -DLLVM_ENABLE_RUNTIMES="openmp" \
        -DLLVM_TARGETS_TO_BUILD="Native" \
        -DLLVM_USE_LINKER=lld \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DMLIR_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_DOCS=OFF \
        -DMLIR_INCLUDE_DOCS=OFF \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DLLVM_LINK_LLVM_DYLIB=ON \
        -DCLANG_LINK_CLANG_DYLIB=ON \
        -DMLIR_LINK_MLIR_DYLIB=ON && \
    ninja && ninja install

# strip the llvm binaries to reduce the image size
RUN find ${INSTALL_PREFIX}/bin ${INSTALL_PREFIX}/lib -type f -exec sh -c '\
      for f do \
        if file "$f" | grep -q "ELF"; then \
          strip --strip-unneeded "$f" || true; \
        fi; \
      done' sh {} +

FROM ubuntu:24.04 AS llvm-base

ARG INSTALL_PREFIX=/opt/llvm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgcc-s1 libc6 libgomp1 python3 && \
  rm -rf /var/lib/apt/lists/*

COPY --from=llvm-builder ${INSTALL_PREFIX} ${INSTALL_PREFIX}

ENV PATH="${INSTALL_PREFIX}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib"
ENV CMAKE_PREFIX_PATH="${INSTALL_PREFIX}"
ENV LLVM_DIR="${INSTALL_PREFIX}/lib/cmake/llvm"
ENV MLIR_DIR="${INSTALL_PREFIX}/lib/cmake/mlir"
ENV LLVM_BASE_DIR="${INSTALL_PREFIX}"

CMD ["/bin/bash"]
