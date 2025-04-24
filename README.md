## Usage Instructions

### Build and Run Container 

Start the container with the default command (runs cuda_test.py):

```bash
docker compose -f compose-nvidia.yml up
```

### Interactive Development Shell

Launch an interactive bash session with GPU support:

```bash
docker compose -f compose-nvidia.yml run --rm nvidia-pytorch bash
```

Once inside the container, test GPU functionality:

```bash
python cuda_test.py
```

Train models
```bash
python train.py --config ./config/config_1.json
```

Test models
```bash
python test.py --config ./config/config_7.json
```

Visualize models
```bash
python visual_run.py --config ./config/config_7.json
```
