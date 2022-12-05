# PLSNet

## Dataset

### ABIDE

Please follow the [instruction](util/abide/readme.md) to download and process this dataset.

## Usage

### ABIDE 

```bash
python main.py --config_filename setting/abide_RGTNet.yaml
```

## Hyper parameters

All hyper parameters can be tuned in setting files.

```yaml
model:
  type: PLSNet
  extractor_type: attention
  embedding_size: 8
  window_size: 4

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
  
  # uniform or pearson
  pure_gnn_graph: pearson
```

# Model Zoo
We provide models for RGTNet_AAL and RGTNet_CC200.

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>atlas</th>
      <th>acc.%</th>
      <th>sen.%</th>
      <th>spe.%</th>
      <th>url of model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAL</td>
      <td>72.4</td>
      <td>71.6</td>
      <td>71.3</td>
      <td><a href="https://pan.baidu.com/s/1K_yXsK0n01mtD1-drTqv8w">baidu disk</a>&nbsp;(code: 7fig)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CC200</td>
      <td>76.4</td>
      <td>74.7</td>
      <td>78.6</td>
      <td><a href="https://pan.baidu.com/s/1apwl5TAzrQbp8wWLdH-BLg">baidu disk</a>&nbsp;(code: pmbz)</td>
    </tr>

  </tbody>
</table>

