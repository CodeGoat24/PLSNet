# PLSNet

The whole repository will be completed soon.

## Dataset


### ABIDE

### Usage

```
cd util/abide/

python 01-fetch_data.py --root_path /path/to/the/save/folder/ --id_file_path subject_IDs.txt --download True

python 02-process_data.py --root_path /path/to/the/save/folder/ --id_file_path subject_IDs.txt

python 03-generate_abide_dataset.py --root_path /path/to/the/save/folder/
```

### Train 

```bash
python main.py --config_filename setting/abide_PLSNet.yaml
```

## Hyper parameters

All hyper parameters can be tuned in setting files.

```yaml
data:
  dataset: ABIDE
  atlas: aal
  batch_size: 16
  test_batch_size: 16
  val_batch_size: 16
  train_set: 0.7
  val_set: 0.1
  fold: 0
  time_seires: {your path}/abide.npy

model:
  type: PLSNet
  extractor_type: attention
  embedding_size: 8

  dropout: 0.5

train:
  lr: 1.0e-4
  weight_decay: 1.0e-4
  epochs: 500
  pool_ratio: 0.7
  optimizer: adam
  stepsize: 200

  group_loss: true
  sparsity_loss: true
  sparsity_loss_weight: 0.5e-4
  log_folder: result
```



