CKPT=/home/s/tuyenld/mae/runs/pretrain/exp22/weight/last.pth
ENCODER=/home/s/tuyenld/mae/ckpt/exp22/encoder.pth
DECODER=/home/s/tuyenld/mae/ckpt/exp22/decoder.pth

python tool/split_encoder_decoder.py --ckpt $CKPT --encoder $ENCODER --decoder $DECODER
