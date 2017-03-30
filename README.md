# TCNActionRecognition
Skeleton based action recognition models with TCN variants for learning interpretable representation.

### NTURGDB Skeleton Cross Subject Validation
Model | Augment | Training Loss | Testing Loss | Validation Acc | Depth | Filter Dim(s) | Layer Widths | Dropout | Opti | SLURM ID|Notes
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---: | :---: | :---: |--- |
Vanila tktcn | 0 | 0.7460 | 2.0484 | 0.639 | 3 | 25 | {128,256,512} | 0.5 | SGD | | |
Vanila tktcn_aug_dr30 | 1 | 1.4947 | 1.7037 | 0.6012 | 3 | 25 | {128,256,512} | 0.3 | SGD | 13843851| try with no droupout, larger LR |
Vanila tktcn_aug_dr50 | 1 | 1.6743 | 1.6113 | 0.615 | 3 | 25 | {128,256,512} |0.5| SGD | |keep bigger LR for longer
Vanila tktcn_gap | 0 | 0.8592 | 1.9153 | 0.636 | 3 | 25 | {128,256,512} | 0.5 | SGD |13860091 | |
tktcn+multiscale_dr50 | 0 | 0.7749 | 1.9637 | 0.635 | 3 | {8,16} | {128,256,512} | 0.5 |SGD| |
tktcn+multiscale_dr30 | 0 | 0.5926 | 2.2443 | 0.622 | 3 | {8,16} | {128,256,512} | 0.3 |SGD| 13859444 |
tktcn+multiscale_dr50_gap | 0 | 0.7421 | 2.013 | 0.642 | 3 | {8,16} | {128,256,512} | 0.5 |SGD| 13869040 |
tktcn+resnet9_f25 |0 | 0.4664 | 2.3786| 0.578 | 9 | 25 | {64x3,128x3,256x3} | 0.0 |SGD| 13858800 |
tktcn+resnet9 | 0 | 0.3269 | 2.7140 | 0.575 | 9 | 8 | {64x3,128x3,256x3} | 0.0 |SGD| 13858891 |
tktcn+resnet9_dropout | 0 | 0.5551 | 2.4392 | 0.577| 9 | 8 | {64x3,128x3,256x3} | 0.5 |SGD| 13858979 |
tktcn+resnet9_gap | 0 | 0.6109 | 1.4615| 0.720 | 9 | 8 | {64x3,128x3,256x3} | 0.0 |SGD| 13859174 |
**tktcn+resnet9_gap_dropout** | 0 | 0.8100 | 1.3288 | **0.727** | 9 | 8 | {64x3,128x3,256x3} | 0.5 |SGD| 13859234 |
tktcn+resnet9_gap_aug | 1 | 0.8604 | 1.3948 | 0.723 | 9 | 8 | {64x3,128x3,256x3} | 0.0 |SGD| 13951341 |
tktcn+resnet9_gap_aug_dropout | 1 | 1.400 | 1.3478 | 0.727 | 9 | 8 | {64x3,128x3,256x3} | 0.3 |SGD| 14127062 |
tktcn+resnet18_gap_dropout | 0 | 0.8698 | 1.3130 | 0.722 | 18 | 8 | {32x6,64x6,128x6} | 0.5 |SGD| 13879610 |
tktcn+resnet9_gap_dropout_tanh | 0 | 1.2466 | 1.4405| 0.681 | 9 | 8 | {64x3,128x3,256x3} | 0.5 |SGD| 13860275 |
tktcn+resnet18_v2_gap | 0 | 0.8384 | 1.5117| 0.711 | 18 | 8 | {32x2x3,64x2x3,128x2x3} | 0.0 |SGD| 13869079 |
tktcn+resnet9_v2.2_gap_dropout | 0 | 0.9105 | 1.4466| 0.705 | 9 | 8 | {64x3,128x3,256x3} | 0.5 |SGD| 13882217 |
tktcn+resnet9_v3_gap | 0 | 0.7708 | 1.5179 | 0.709 | 9 | {8,16} | {32x2x3,64x2x3,128x2x3} | 0.0 |SGD| 13870423 |
tktcn+resnet9_v3_gap_dropout | 0 | 0.9673 | 1.4405 | 0.677 | 9 | {8,16} | {32x2x3,64x2x3,128x2x3} | 0.5 |SGD| 13885555 |


### NTURGDB Skeleton Cross View Validation
Model | Augment | Training Loss | Testing Loss | Validation Acc | Depth | Filter Dim(s) | Layer Widths | Dropout | Opti | SLURM ID|Notes
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---: | :---: | :---: |--- |
Vanila tktcn | 0 | 0.8279 | 1.0848 | 0.805 | 3 | 25 | {128,256, 512} | 0.5 |SGD| 14184708 |
**tktcn+resnet9_gap** | 0 | 0.6173 | 1.0405 | **0.819** | 9 | 8 | {64x3,128x3,256x3} | 0.0 |SGD| 14175241 |
tktcn+resnet9_gap_dropout | 0 | 0.7375 | 1.0234 | 0.811 | 9 | 8 | {64x3,128x3,256x3} | 0.5 |SGD| 14174425 |


### NTURGDB Skeleton Cross Subject Raw Validation
Model | Augment | Training Loss | Testing Loss | Validation Acc | Depth | Filter Dim(s) | Layer Widths | Dropout | Opti | SLURM ID|Notes
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---: | :---: | :---: |--- |
**tktcn_resnet9** | 0 | 0.8401 | 1.2822 | **0.743** | 9 | 8 | {64x3,128x3,256x3} | 0.5 |SGD| 14239352 |
tktcn_resnet9_m0 | 0 | x | x | x | 9 | 8 | {64x3,128x3,256x3} | 0.5 |SGD| 519 |

### NTURGDB Skeleton Cross View Raw Validation
Model | Augment | Training Loss | Testing Loss | Validation Acc | Depth | Filter Dim(s) | Layer Widths | Dropout | Opti | SLURM ID|Notes
---|:---:|:---:|:---:|:---:|:---:|:---:|:---: | :---: | :---: | :---: |--- |
**tktcn_resnet9** | 0 | 0.8839 | 0.8993 | **0.831** | 9 | 8 | {64x3,128x3,256x3} | 0.5 |SGD| 15353 |


^ : difference between resnet versions
  - resnet: one convolution per residual block  (one wider conv block)
  - resnet_v2: two convolutions per residual block (two stacked narrower conv blocks)
  - resnet_v2.2: three convolutions per residual block (three stacked same conv blocks)
  - resnet_v3: two convolutions of different filter lengths per residual block, not stacked (inception+resnet)
  - resnet_v4: no relu, only visualization focus with tanh, BN+nonlin after merge, resblock only contains conv


