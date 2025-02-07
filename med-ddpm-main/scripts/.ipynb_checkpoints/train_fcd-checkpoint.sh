python3 ../train.py --fcd_dataset --with_condition --inputfolder ../../data/pathological_mri/ --results_folder ./results/train_fcd_data --batchsize 1 --epochs 10001 --input_size 128 --depth_size 128 --num_channels 64 --num_res_blocks 1 --timesteps 250 --save_and_sample_every 100  --resume_weight ../model/model_128.pt
