save_path="../../results/bee_dance_kn_xscpe/"
bw=0.5
data_difference=1
epsilon=-5
eval_test_every=1
gpu=0
lambda_cov=-1
lr=-6
min_dist=1
num_iters=100
num_layers=5
nums_labeled=(0 1 2 3 4)
seed=0
window_size=3

for seed in $(seq 3 3 30)
do
    for num_labeled in "${nums_labeled[@]}"
    do
        python bee_dance_kn_xscpe.py --bw $bw --data_difference $data_difference --epsilon $epsilon \
        --eval_test_every $eval_test_every --gpu $gpu --lambda_cov $lambda_cov --lr $lr --min_dist $min_dist \
        --num_iters $num_iters --num_labeled $num_labeled --num_layers $num_layers --save_path $save_path --seed $seed \
        --window_size $window_size
    done
done
