# titansoft_deep_sad

## Ocsvm

```shell
mkdir log/ocsvm/
mkdir log/ocsvm/titansoft
cd src

python3 baseline_ocsvm.py titansoft ../log/ocsvm/titansoft ../data --ratio_known_outlier 0 --ratio_pollution 0 --kernel rbf --normal_class 0 --known_outlier_class 0 --n_known_outlier_classes 0 --seed 0;
```
```
INFO:root:Log file is ../log/ocsvm/titansoft/log.txt.
INFO:root:Data path is ../data.
INFO:root:Export path is ../log/ocsvm/titansoft.
INFO:root:Dataset: titansoft
INFO:root:Normal class: 0
INFO:root:Ratio of labeled normal train samples: 0.00
INFO:root:Ratio of labeled anomalous samples: 0.00
INFO:root:Pollution ratio of unlabeled train data: 0.00
INFO:root:Number of known anomaly classes: 0
INFO:root:OC-SVM kernel: rbf
INFO:root:Nu-paramerter: 0.10
INFO:root:Hybrid model: False
INFO:root:Set seed to 0.
INFO:root:Computation device: cpu
INFO:root:Number of dataloader workers: 0
INFO:root:Starting training...
INFO:root:  | Model 01/10 | Gamma: 0.00781250 | Train Time: 1.424s | Val AUC: 68.97 |
INFO:root:  | Model 02/10 | Gamma: 0.01562500 | Train Time: 1.451s | Val AUC: 69.08 |
INFO:root:  | Model 03/10 | Gamma: 0.03125000 | Train Time: 1.397s | Val AUC: 69.32 |
INFO:root:  | Model 04/10 | Gamma: 0.06250000 | Train Time: 1.444s | Val AUC: 69.94 |
INFO:root:  | Model 05/10 | Gamma: 0.12500000 | Train Time: 1.393s | Val AUC: 70.47 |
INFO:root:  | Model 06/10 | Gamma: 0.25000000 | Train Time: 1.347s | Val AUC: 71.67 |
INFO:root:  | Model 07/10 | Gamma: 0.50000000 | Train Time: 1.446s | Val AUC: 73.17 |
INFO:root:  | Model 08/10 | Gamma: 1.00000000 | Train Time: 1.540s | Val AUC: 73.95 |
INFO:root:  | Model 09/10 | Gamma: 2.00000000 | Train Time: 1.680s | Val AUC: 74.24 |
INFO:root:  | Model 10/10 | Gamma: 4.00000000 | Train Time: 1.460s | Val AUC: 74.88 |
INFO:root:Best Model: | Gamma: 4.00000000 | AUC: 74.88
INFO:root:Training Time: 1.460s
INFO:root:Finished training.
INFO:root:Starting testing...
INFO:root:Test AUC: 67.95%
INFO:root:Test Time: 0.327s
INFO:root:Finished testing.

roc_auc_score 0.6794565914726128
===best===
max_f1 0.3856041131105398
m_precision 0.2640845070422535
m_recell 0.7142857142857143
threshod -6.406195381162039
acc 0.6078293483356775
TN,FP,FN,TP= 2068.0 1463.0 210.0 525.0
```



## Deep SAD 
```shell
### create folders for experimental output
mkdir log/DeepSAD
mkdir log/DeepSAD/titansoft

#### change to source directory
cd src

#### run experiment
python3 main.py titansoft titansoft_mlp ../log/DeepSAD/titansoft ../data --ratio_known_outlier 0.01 --ratio_pollution 0.1 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1 --seed 0;
```

```
INFO:root:Test AUC: 65.68%
INFO:root:Test AUC: 67.68%

eval time:  1.2488980293273926
roc_auc_score 0.6768044742677587
===best===
max_f1 0.40418118466898956
m_precision 0.4142857142857143
m_recell 0.3945578231292517
threshod 0.15477372705936432
acc 0.79957805907173
TN,FP,FN,TP= 3121.0 410.0 445.0 290.0
```



python3 main.py titansoft titansoft_mlp ../log/DeepSAD/titansoft ../data --ratio_known_outlier 0 --ratio_pollution 0 --lr 0.0001 --n_epochs 50 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 50 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 0 --seed 0;
