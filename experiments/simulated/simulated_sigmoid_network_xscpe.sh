save_path="../../results/simulated_sigmoid_network_xscpe/"
epsilon=-4
eval_test_every=1
gpu=0
lambda_cov=4
lambda_params=-10
lr=-3
min_dist=1
num_iters=100
nums_labeled=(0 1 2 4 8 16 32 64 128 256)
std=-2

for seed in $(seq 3 3 30)
do
    for num_labeled in "${nums_labeled[@]}"
    do
        python simulated_sigmoid_network_xscpe.py --epsilon $epsilon --eval_test_every $eval_test_every --gpu $gpu \
        --lambda_cov $lambda_cov --lambda_params $lambda_params --lr $lr --min_dist $min_dist --num_iters $num_iters \
        --num_labeled $num_labeled --save_path $save_path --seed $seed --std $std
    done
done
