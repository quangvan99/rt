cmake -S . -B build -D TRT_PATH=/usr/lib/x86_64-linux-gnu/ -D BUILD_PYTHON=ON -D CMAKE_INSTALL_PREFIX=install/
cmake --build build -j$(nproc) --config Release --target install