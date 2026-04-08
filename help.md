export INPUT_PATH="/mnt/c/Users/munte/Downloads/kth_organized/"



rm -rf *
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_GUI=NO
make -j$(nproc)

python3 ../src/tool/extract_bboxes_kth.py
nohup ./KTH_3D > log_kth.txt 2>&1 &
tail -f log_kth.txt

----------------------------------------------------------------

Este perfect normal! Când se întâmplă asta și vrei să lucrezi din nou, tot ce ai de făcut este să repeți rutina pe care abia ai învățat-o:
Dai iar oarsub în terminal.
Primești un nod nou (să zicem roazhon13-7).
Te duci în fișierul config și schimbi numele de la Host și HostName.
Te duci în CLion la Toolchains -> Credentials și pui noul nume. Și gata, ești înapoi în afaceri în mai puțin de 1 minut!

ssh rennes.grid5000.fr
oarsub -I -p "cluster='roazhon13'" -l walltime=5:00:00

## scp -r C:\Users\munte\Downloads\kth_organized rennes.grid5000.fr:~/

## Iarasi de rulat:
sudo-g5k apt-get update
sudo-g5k apt-get install -y libopencv-dev
sudo-g5k apt-get update
sudo-g5k apt-get install -y liblapacke-dev liblapack-dev libblas-dev
sudo-g5k apt-get update
sudo apt-get install libopenblas-dev 
