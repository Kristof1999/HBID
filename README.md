# HBID
Hybrid Blind Image Deconvolution

The code is based on [SelfDeblur](https://github.com/csdwren/SelfDeblur).

Borrowes ideas from: VPNet [^1], MAPk [^2], and SelfDeblur [^3].

## Running

You can see the available variants in the variants folder. To run one of them, go to the experiment.py file and uncomment it, then run the experiment.py script.

Output will be available in results/levin/{variant name} folder.

[^1]: VPNet: Variable Projection Networks, Péter Kovács and Gergő Bognár and Christian Huber and Mario Huemer, 2022
[^2]: Efficient Marginal Likelihood Optimization in Blind Deconvolution, Anat Levin and Yair Weiss and Fredo Durand and William T. Freeman, 2011
[^3]: Neural Blind Deconvolution Using Deep Priors, Dongwei Ren and Kai Zhang and Qilong Wang and Qinghua Hu and Wangmeng Zuo, 2020
