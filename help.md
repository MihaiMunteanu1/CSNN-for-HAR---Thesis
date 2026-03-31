export INPUT_PATH="/mnt/c/Users/munte/Downloads/kth_organized/"



rm -rf *
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_GUI=NO
make -j$(nproc)

nohup ./KTH_3D > log_kth.txt 2>&1 &
tail -f log_kth.txt