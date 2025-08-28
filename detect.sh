g++ -std=c++17 -O2 detect.cpp -o detect \
    `pkg-config --cflags --libs opencv4` \
    -I./install/include \
    -L./install/lib -ltrtyolo -lcustom_plugins \
    -Wl,-rpath=./install/lib

./detect ../models/yolov8n_backbone.engine ../imgs/img.jpg