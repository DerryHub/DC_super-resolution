# 4倍模型记录

|  模型  |    优化方法    | 损失函数 | EPOCH | 测试时长(s) | PSNR  | SSIM | 备注             |
| :----: | :------------: | :------: | :---: | :---------: | :---: | :--: | ---------------- |
|  CARN  |   adam/1e-4    | MSELoss  |  20   |     46      | 27.92 | 0.81 | loss还有下降空间 |
| CARN M |   adam/1e-4    | MSELoss  |  20   |     46      | 28.06 | 0.82 | loss还有下降空间 |
|  CARN  |   adam/1e-4    | MSELoss  |  50   |     46      | 28.4  | 0.83 | loss基本饱和     |
| CARN M |   adam/1e-4    | MSELoss  |  50   |     46      | 28.2  | 0.82 | loss基本饱和     |
|  EDSR  |   adam/1e-4    | MSELoss  |  20   |     69      | 28.2  | 0.82 | loss还有下降空间 |
|  CARN  |   adam/1e-4    |  L1Loss  |  20   |     46      | 28.22 | 0.82 | loss基本饱和     |
| CARN M |   adam/1e-4    |  L1Loss  |  20   |     46      | 28.12 | 0.82 | loss基本饱和     |
|  EDSR  |   adam/1e-4    |  L1Loss  |  20   |     69      | 28.11 | 0.82 | loss基本饱和     |
|  CARN  | adam/1e-4/1e-5 |  L1Loss  |  50   |     46      | 28.31 | 0.82 | loss基本饱和     |
|  EDSR  |   adam/1e-4    |  L1Loss  |  50   |     69      | 28.39 | 0.83 | loss基本饱和     |
|  EDSR  |   adam/1e-4    | MSELoss  |  50   |     69      | 28.48 | 0.83 | loss基本饱和     |

# 2倍模型记录

|  模型  | 优化方法  | 损失函数 | EPOCH | 测试时长(s) | PSNR  | SSIM | 备注             |
| :----: | :-------: | :------: | :---: | :---------: | :---: | :--: | ---------------- |
|  CARN  | adam/1e-4 | MSELoss  |  20   |     106     | 33.69 | 0.94 | loss还有下降空间 |
| CARN M | adam/1e-4 | MSELoss  |  20   |     106     | 33.75 | 0.94 | loss还有下降空间 |
|  CARN  | adam/1e-4 | MSELoss  |  50   |     106     | 34.37 | 0.95 | loss基本饱和     |
|  CARN M | adam/1e-4 | MSELoss  |  50   |     106     | 34.10 | 0.95 | loss基本饱和     |
|  EDSR  | adam/1e-4 | MSELoss  |  20   |     167     | 33.90 | 0.94 | loss还有下降空间 |
|  CARN  | adam/1e-4 |  L1Loss  |  20   |     106     | 33.88 | 0.95 | loss基本饱和     |
|  CARN  | adam/1e-4 |  L1Loss  |  20   |     106     | 33.93 | 0.94 | loss基本饱和     |
|  CARN  | adam/1e-4 |  L1Loss  |  20   |     106     | 34.17 | 0.95 | loss基本饱和     |
|  EDSR  | adam/1e-4 |  L1Loss  |  20   |     167     | 34.04 | 0.95 | loss基本饱和     |
|  EDSR  | adam/1e-4 | MSELoss  |  50   |     167     | 34.48 | 0.95 | loss基本饱和     |

