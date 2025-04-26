# Test
```
python3 test.py -m cross_fusion -bb clft -p ./waymo_dataset/splits_clft/test_day_fair.txt
```

# Visual

```
python3 visual_run.py -m cross_fusion -bb clft -p ./waymo_dataset/splits_clft/test_day_fair.txt
python3 visual_run.py -m cross_fusion -bb clft -p ./waymo_dataset/splits_clft/all.txt
```

# Start
```
docker compose -f compose-nvidia.yml run --rm nvidia-pytorch bash
```

# Conda setup
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate

conda env create -f clft_py39_torch21_env.yml
conda env list
conda activate clft_py39_torch21
```
