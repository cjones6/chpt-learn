save_path="../../results/mnist-seq_lenet-5_ckn_xscpe/"
batch_size=50
epsilon=-3
eval_test_every=1
gpu=0
lambda_cov=3
lr=-3
min_dist=1
num_iters=100
nums_labeled=(0 1 2 4 8 16 32 64 128 256)

for seed in $(seq 3 3 30)
do
    for num_labeled in "${nums_labeled[@]}"
    do
        python mnist-seq_lenet-5_ckn_xscpe.py --batch_size $batch_size --epsilon $epsilon \
        --eval_test_every $eval_test_every --gpu $gpu --lambda_cov $lambda_cov --lr $lr --min_dist $min_dist \
        --num_iters $num_iters --num_labeled $num_labeled --save_path $save_path --seed $seed
    done
done
