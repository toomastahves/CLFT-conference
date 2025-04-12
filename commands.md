# train
cd /gpfs/mariana/home/totahv/Projects/CLFT-conference/
git pull
conda activate conda_py39
chmod +x train.sh
sbatch -J train-1 train.slurm ./config/icaart2025/config_7.json
watch -n1 squeue -u totahv

# test
chmod +x test.sh
sbatch -J test-1 test.slurm ./config/icaart2025/config_7.json
python3 test.py --config ./config/icaart2025/config_7.json
