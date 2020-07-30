save_path="../../results/bee_dance_kcpe/"

# KCPE with RBF kernel using a tuned bandwidth
python bee_dance_kcpe.py --bw 0.2 --data_difference 1 --min_dist 1 --save_path $save_path --window_size 3
