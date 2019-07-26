# HARD CODED

# Set id of GPU
g=2

# Create logFile folder
mkdir -p logFiles

# Initialization 4x4
python pgan/4x4.py --gpu $g > logFiles/4x4.txt

# Transition 4 -> 8
python pgan/4x8.py --gpu $g --load=train_log/4x4/checkpoint > logFiles/4x8.txt

# Stabilization 8x8
python pgan/8x8.py --gpu $g --load=train_log/4x8/checkpoint > logFiles/8x8.txt

# Transition 8 -> 16
python pgan/8x16.py --gpu $g --load=train_log/8x8/checkpoint > logFiles/8x16.txt

# Stabilization 16x16
python pgan/16x16.py --gpu $g --load=train_log/8x16/checkpoint > logFiles/16x16.txt

# Transition 16 -> 32
python pgan/16x32.py --gpu $g --load=train_log/16x16/checkpoint > logFiles/16x32.txt

# Stabilization 32x32
python pgan/32x32.py --gpu $g --load=train_log/16x32/checkpoint > logFiles/32x32.txt

# Transition 32 -> 64
python pgan/32x64.py --gpu $g --load=train_log/32x32/checkpoint > logFiles/32x64.txt

# Stabilization 64x64
python pgan/64x64.py --gpu $g --load=train_log/32x64/checkpoint > logFiles/64x64.txt

# Transition 64 -> 128
python pgan/64x128.py --gpu $g --load=train_log/64x64/checkpoint > logFiles/64x128.txt

# Stabilization 128x128
python pgan/128x128.py --gpu $g --load=train_log/64x128/checkpoint > logFiles/128x128.txt

# Transition 128 -> 256
python pgan/128x256.py --gpu $g --load=train_log/128x128/checkpoint > logFiles/128x256.txt

# Stabilization 256x256
python pgan/256x256.py --gpu $g --load=train_log/128x256/checkpoint > logFiles/256x256.txt
