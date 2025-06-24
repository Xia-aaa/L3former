# L³former

Official implementation of the paper:  
**[L³former: Enhanced Multi-Scale Shared Transformer with Local Linear Layer for Long-term Series Forecasting](https://doi.org/10.1016/j.inffus.2025.103398)**


## 📜 Citation
```bash
@article{xia2025L³former,
  title  = {L³former: Enhanced Multi-Scale Shared Transformer with Local Linear Layer for Long-term Series Forecasting},
  author = {Yulin Xia and Chang Wu and Xiaoman Yang},
  journal= {Information Fusion},
  volume = {103398},
  year   = {2025},
  doi    = {10.1016/j.inffus.2025.103398},
  note   = {Available online: 20 June 2025},
  url    = {https://doi.org/10.1016/j.inffus.2025.103398}
}
```

## 🚀 News

-   ​**​2025.06​**​ L³former is officially online and release full code of main results.
-   ​**​2025.05​**​ L³former is completely accepted by **Information Fusion**.

## 🔍Key Features

-   🚀 ​**​Local Linear Layer (L³)​**​: Novel neural network layer capturing fine-grained local-temporal patterns
-   🚀 ​**​Scale-Wise Attention (SWAM)​**​: Efficient multi-scale feature fusion with O(G²) complexity
-   🚀 ​**​Variable-Wise Feed-Forward (VWFF)​**​: Enhanced modeling of inter-variable correlations
-   🚀 ​**​Triple-Observation Framework​**​: Simultaneous modeling of scale, temporal, and variable dependencies
-   🚀 ​**​Multi-Scale Shared Backbone​**​: Unified architecture reducing model complexity and accelerating training
-  🚀 ​**​State-of-the-art performance​**​ in long-term time series forecasting with ​**​5.8%-16.7% lower MSE​**​ averaged by all the datasets compared to previous methods
<p align="center">
  <img src="https://raw.githubusercontent.com/Xia-aaa/L3former/main/figures/Radar.png" width="500"/>
</p>

## 🧠 Core Components
### 1️⃣ Model Structure (L³former)
**Triple-Observation Framework​**​: Simultaneous modeling of scale, temporal, and variable dependencies

<p align="center">
  <img src="https://raw.githubusercontent.com/Xia-aaa/L3former/main/figures/L3former.png" width="700"/>
</p>

### 2️⃣ Local Linear Layer (L³)

Innovative neural layer replacing convolutions/pooling:

​**​Key advantages​**​:
-   ✅ Temporal-independent weights for local temporal modeling, and channel-shared weights for cross-variable interaction
-   ✅ Adjustable window sizes for multi-scale feature capture
-   ✅ 49% Faster and 99.9% Fewer Parameters over 1D convolution for high-dimensional data processing


<p align="center">
  <img src="https://raw.githubusercontent.com/Xia-aaa/L3former/main/figures/L3andConv.png" width="700"/>
</p>

</p>

### 3️⃣ Scale-Wise Attention (SWAM)

Computationally efficient attention across temporal scales:

​**​Key advantages​**​:

-   ✅ Scale number G (≤6) is independent of sequence length and variable numbers, reducing attention computation cost
-   ✅ Effectively fuse cross-scale features

<p align="center">
  <img src="https://raw.githubusercontent.com/Xia-aaa/L3former/main/figures/SWAM.png" width="500"/>
</p>

### 4️⃣ Variable-Wise Feed-Forward (VWFF)

Captures complex cross-variable dependencies:

​**​Key advantages​**​:

-   ✅ Integrate series-wise embedding to overcome VWFF failure.
-   ✅ Linear complexity with low cost

<p align="center">
  <img src="https://raw.githubusercontent.com/Xia-aaa/L3former/main/figures/VWFF.png" width="700"/>
</p>

##  📊 Scability of L³former

-    ​**VWFF+PatchTST**​: Table 4 in our paper
-   **​L³+iTransformer​**​: Table 6 in our paper
-  **​L³+TimeMixer**​: Table 7 in our paper

##  📊 Experienments
### 1️⃣ Main Results
Table 2 in our paper: 
L³former achieves significant improvements in long-term forecasting across nine benchmark datasets, outperforming state-of-the-art models (in Fig.1) by **5.8%–16.7%**  averaged over all datasets in MSE.
- **Large-Scale Datasets**： On **Solar-Energy**, L³former reduces MSE by **11.1%** compared to the best baseline; on **ECL**, it lowers MSE by **7.9%**. On **Traffic**, while MSE is slightly higher than iTransformer, MAE remains comparable. 
-  **Other Datasets**：On **Weather**, L³former improves MSE by **2.6%–4.1%**, and on **ETTm** datasets, it surpasses all baselines, including linear-based models like TimeMixer. 
-  **Ultra-long-term forecasting**：In 720-step prediction, L³former demonstrates robust performance: it reduces MSE by **13.5% (ECL)**, **11.8% (Solar-Energy)**, and **26.3% (Exchange)**，
### 2️⃣ Ablation
Table 8 in our paper
### 3️⃣ Local Linear Layer (L³)
✅ Compared with pooling layer and convolution layer
<p align="center">
  <img src="https://raw.githubusercontent.com/Xia-aaa/L3former/main/figures/L3vsPoolingvsConv.png" width="700"/>
</p>
✅ Different windows
<p align="center">
  <img src="https://raw.githubusercontent.com/Xia-aaa/L3former/main/figures/Windows.png" width="400"/>
</p>

### 4️⃣ VWFF and Temporal shifts in the Traffic dataset
The performance drop of VWFF on the Traffic dataset stems from the **840th anomalous variable** causing model misguidance and temporal drift; removing it restores accuracy, proving VWFF’s robustness in stable-variable scenarios (e.g., ECL, Weather). For Traffic-like data, we suggest to **apply rigorous variable screening or anomaly detection** before deployment.
<p align="center">
  <img src="https://raw.githubusercontent.com/Xia-aaa/L3former/main/figures/Traffic.png" width="600"/>
</p>

### 5️⃣ Efficiency
L³former demonstrates significantly lower GPU memory usage and faster computational speed than multi-scale Transformers (e.g., **Pathformer, Crossformer**), even outperforming the linear model **TimeMixer**.
<p align="center">
  <img src="https://raw.githubusercontent.com/Xia-aaa/L3former/main/figures/Efficiency.png" width="700"/>
</p>

### 6️⃣ Other experiements
Comprehensive experiments were conducted. For additional details, please refer to the original paper.

## 🚂 Getting Started

### Installation
```bash
cd L3former-main
pip install -r requirements.txt
```

### Data Preparation
The datasets can be obtained from  [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link)  or  [Baidu Cloud](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr).

### Training & Evaluation

Run benchmark experiments:
```bash
# for example: ECL
bash scripts/multivariate_forecasting/ECL.sh
# for example: ETTh1
bash scripts/multivariate_forecasting/ETTh1.sh
```
The you can cheak the results in "./logs". Moreover, we have provided the complete training logs in "./logs/2025".

## 🙏 Acknowledgements

We extend gratitude to these foundational works:

-   Informer ([https://github.com/zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020))
-   Autoformer ([https://github.com/thuml/Autoformer](https://github.com/thuml/Autoformer))
- PatchTST ([https://github.com/yuqinie98/PatchTST.](https://github.com/yuqinie98/PatchTST))
-   Time-Series-Library ([https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library))
-   iTransformer ([https://github.com/thuml/iTransformer.](https://github.com/thuml/Time-Series-Library))
- TimeMixer ([https://github.com/kwuking/TimeMixer.](https://github.com/kwuking/TimeMixer))
## 📧 Contact

Reach our team for collaboration opportunities:

-   Yulin Xia: xyrain1@std.uestc.edu.cn
