save_path="../../results/final/simulated_sigmoid_network_kcpe/"
bw=1.0
std=-2

for seed in $(seq 3 3 30)
do
    python simulated_sigmoid_network_kcpe.py --bw $bw --save_path $save_path --seed $seed --std $std
done

