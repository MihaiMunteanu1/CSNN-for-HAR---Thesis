Workflow nou pe VM (în ordine)

# 1. Reorganizeaza folderul (o singura data)
python3 cnn_har_app/kth_reorganize_splits.py \
--input  /home/mmuntean/kth_organized \
--output /home/mmuntean/kth_organized_tvt

# 2. Regenereaza bbox JSON (paths cu prefixul tvt)
python3 src/tool/extract_bboxes_kth.py
# -> ../hog/hog_person_data_tvt_7_g4.json

# 3a. Pentru CSNN full-frame + HOGSampler3D
python3 extract_full_frames_kth.py \
--bbox_json ../hog/hog_person_data_tvt_7_g4.json \
--output ../hog/kth_fullframes_tvt_g4 \
--video_root /home/mmuntean/kth_organized_tvt

# 3b. Pentru CSNN crop legacy (daca vrei comparatie)
python3 cnn_har_app/extract_frames_kth.py \
--bbox_json ../hog/hog_person_data_tvt_7_g4.json \
--output ../hog/kth_frames_tvt_g4 \
--video_root /home/mmuntean/kth_organized_tvt

# 3c. Pentru CNN
python3 cnn_har_app/extract_hog_augmented.py \
--bbox_json ../hog/hog_person_data_tvt_7_g4.json \
--output ../hog/hog_aug_tvt_g4.npz \
--num_aug 8 \
--aug_profile strong \
--video_root /home/mmuntean/kth_organized_tvt





----------------------------------------------------------------

## Iarasi de rulat:
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_GUI=NO
make -j$(nproc)
sudo-g5k apt-get update
sudo-g5k apt-get install -y libopencv-dev liblapacke-dev liblapack-dev libblas-dev libopenblas-dev

+WALLTIME:
oarwalltime 281653 +02:00:00

----------------------------------------------

1. Oprește instalarea curentă dacă încă rulează
# Ctrl+C în terminalul unde rulează pip install

2. Curăță orice s-a instalat din greșeală în ~/.local
   rm -rf ~/.local/lib/python3.9/site-packages/torch*
   rm -rf ~/.local/lib/python3.9/site-packages/nvidia*
   rm -rf ~/.local/lib/python3.9/site-packages/triton*
   rm -rf ~/.local/lib/python3.9/site-packages/functorch*
   quota -s
# verifică să fie ~9 GB folosit, fără asterisc

3. Șterge venv-ul vechi (poate e corupt de la prima încercare) și fă-l curat
   rm -rf /tmp/mmuntean/venv
   python3 -m venv /tmp/mmuntean/venv

4. ACTIVEAZĂ venv-ul ← pasul critic, ăsta lipsea
   source /tmp/mmuntean/venv/bin/activate
   Acum prompt-ul ar trebui să arate așa:
   (venv) mmuntean@fluxembourg:~$

5. Verifică că folosești pip-ul din venv, NU cel global
   which pip
# OK:   /tmp/mmuntean/venv/bin/pip
# RĂU:  /home/mmuntean/.local/bin/pip  sau /usr/bin/pip
Dacă nu arată /tmp/mmuntean/venv/bin/pip, oprește-te și sună-mă.

6. Instalează pachetele
7. pip uninstall -y triton
   python3 -m pip install opencv-python-headless
   pip install ultralytics
   pip install numpy scikit-learn
   pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3


-----------------------

