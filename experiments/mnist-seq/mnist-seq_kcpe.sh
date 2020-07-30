save_path="../../results/mnist-seq_kcpe/"

for seed in $(seq 3 3 30)
do
    python mnist-seq_kcpe.py --bw 100 --min_dist 1 --save_path $save_path --seed $seed
done
