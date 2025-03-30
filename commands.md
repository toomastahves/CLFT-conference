# train
cd /gpfs/mariana/home/totahv/Projects/CLFT-conference/
git pull
conda activate conda_py39
chmod +x train.sh
sbatch -J train-1 train.slurm ./config/iros2025/config_1.json
sbatch -J train-2 train.slurm ./config/iros2025/config_2.json
sbatch -J train-3 train.slurm ./config/iros2025/config_3.json
watch -n1 squeue -u totahv

# test
chmod +x test.sh
sbatch -J test-1 test.slurm ./config/iros2025/config_1.json
sbatch -J test-2 test.slurm ./config/iros2025/config_2.json
sbatch -J test-3 test.slurm ./config/iros2025/config_3.json
python3 test.py --config ./config/iros2025/config_2.json
