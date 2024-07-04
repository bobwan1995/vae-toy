CUDA_VISIBLE_DEVICES=0

python train.py \
  --epochs 20 \
  --train_batch_size 64 \
  --test_batch_size 128 \
  --lr 1e-3 \
  --exp_name 'lr1e3_im128_h128' \
  --image_size 128 \
  --latent_dim 128