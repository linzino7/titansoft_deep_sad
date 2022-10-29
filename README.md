# titansoft_deep_sad

This Reop. is fork from [Deep SAD](https://github.com/lukasruff/Deep-SAD-PyTorch)

## modify
```
        new file:   src/base/titansoft_dataset.py
        new file:   src/datasets/titansoft.py
        
        modified:   src/baseline_ocsvm.py
        modified:   src/baseline_ssad.py
        modified:   src/datasets/main.py
        modified:   src/main.py
        modified:   src/networks/main.py


```

## preprosess

```
python3 titansoft_deep_sad.py
```

##
Please put preprocessed data to data/titansoft/

## Ocsvm

```shell
mkdir log/ocsvm/
mkdir log/ocsvm/titansoft
cd src

python3 baseline_ocsvm.py titansoft ../log/ocsvm/titansoft ../data --ratio_known_outlier 0 --ratio_pollution 0 --kernel rbf --normal_class 0 --known_outlier_class 0 --n_known_outlier_classes 0 --seed 0;
```
```
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


## Deep SAD with not lable
```
python3 main.py titansoft titansoft_mlp ../log/DeepSAD/titansoft ../data --ratio_known_outlier 0 --ratio_pollution 0 --lr 0.0001 --n_epochs 50 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 50 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 0 --seed 0;
```

```
 python3 measueAUC_pool.py ../log/DeepSAD/titansoft 
```

```
eval time:  0.25827789306640625
roc_auc_score 0.6171732584282652
===best===
max_f1 0.3348325837081459
m_precision 0.2646129541864139
m_recell 0.4557823129251701
threshod 0.0007266211323440075
acc 0.6879981247069855
TN,FP,FN,TP= 2600.0 931.0 400.0 335.0

```


