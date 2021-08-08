readme.md

# 

The Pytorch implementation for the NeurIPS2019 paper of "[Residual Feedback Network for Breast Lesion Segmentation in Ultrasound Image]()".

### Citation

If you find our project useful in your research, please cite it as follows:

```
@inproceedings{kobayashi2019neurips,
  title={Residual Feedback Network for Breast Lesion Segmentation in Ultrasound Image},
  author={Ke Wang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention.},
  year={2021}
}
```

## Contents

1. [Introduction](#introduction)
2. [Dependencies](#Dependencies)
3. [Results](#results)

## Introduction

In this paper, we proposed a novel residual feedback network, which enhances the confidence of the inconclusive pixels to boost breast lesion segmentation performance. 
In the proposed network, a residual representation module is introduced to learn the residual representation of missing/ambiguous boundaries and confusing regions, which promotes the network to make more efforts on those hardly-predicted pixels. 
Moreover, a residual feedback transmission strategy is designed to update the input of the encoder blocks by combining the residual representation with original features. This strategy could enhance the regions including hardly-predicted pixels, which makes the network can further correct the errors in initial segmentation results. 
Experimental results on three datasets (3813 images in total) demonstrate that our proposed network outperforms the state-of-the-art segmentation methods.
For the more detail, please refer to our paper.

<!-- <img width=400 src="https://user-images.githubusercontent.com/53114307/67915023-26daf380-fbd5-11e9-8152-9089b910234d.png"> -->
<img width=400 src="https://github.com/mniwk/RF-Net/blob/main/imgs/Figure%201.png">

Figure: The overview of our proposed network for lesion segmentation in BUS images.


### Dependencies

- [Python3](https://www.python.org/downloads/)
- [PyTorch(>=1.0.0)](http://pytorch.org)

## Results
wo -- without pretrain parameters
w -- with pretrain parameters
| Network  |  DSC | HD |
|---|---|---|
| U-Net [1]|					83.42±0.65 | 14.44±1.98 |
| DeepLabV3^{wo} [2]| 	83.80±0.91 | 8.50±0.89 |
| DeepLabV3^{w} [2]|	85.20±0.53 | 8.13±0.96 |
| CE-Net^{wo} [3]|		84.17±0.55 | 8.01±0.40 |
| CE-Net^{wo} [3]|		86.13±0.43 | 7.58±0.55 |
| S^2P-Net [4]|		84.01±0.35 | 13.63±0.95 |
| Our method |			87.00±0.23 | 6.77±0.51 |

## Reference
1. Ronneberger, et al.: U-Net: convolutional networks for biomedical image segmentation. In: Navab, N., Hornegger, J., Wells, W.M., Frangi, A.F. (eds.) MICCAI 2015. LNCS, vol. 9351, pp. 234-241. Springer, Cham (2015)
2. Chen, L.C., et al.: Rethinking atrous convolution for semantic image segmentation. preprint arXiv:1706.05587 (2017)
3. Gu, Z., et al.: Ce-net: Context encoder network for 2d medical image segmentation. IEEE transactions on medical imaging 38-10, 2281-2292 (2019)
4. Zhu, L., et al.: A Second-Order Subregion Pooling Network for Breast Lesion Segmentation in Ultrasound. In: Anne L., Purang A., Danail S. et al. (eds.) MICCAI 2020. LNCS, vol.12266, pp.160-170. Springer, Cham (2020)
