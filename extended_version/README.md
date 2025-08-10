# GenLV 
### Exploring Scalable Unified Modeling for General Low-Level Vision [[Paper Link]](http://arxiv.org/abs/2507.14801)

Xiangyu Chen*, Kaiwen Zhu*, Yuandong Pu*, Shuo Cao, Xiaohui Li, Wenlong Zhang, Yihao Liu, Yu Qiao, Jiantao Zhou and Chao Dong

### Quick Start
+ Environment
  + pytorch>=1.7
  + basicsr==1.4.2
+ Weights
  + Download the weights from [this link](https://huggingface.co/Kaiwen-Zhu/GenLV/tree/main/GenLV-101/ckpt) and put them in `extended_version/ckpt`.

### How to Inference
```sh
cd extended_version
python inference.py --model_size huge --input example/input.png --prompt_input example/prompt_input.png --prompt_target example/prompt_output.png
```

### Citation

    @article{chen2025exploring,
      title={Exploring Scalable Unified Modeling for General Low-Level Vision},
      author={Chen, Xiangyu and Zhu, Kaiwen and Pu, Yuandong and Cao, Shuo and Li, Xiaohui and Zhang, Wenlong and Liu, Yihao and Qiao, Yu and Zhou, Jiantao and Dong, Chao},
      journal={arXiv preprint arXiv:2507.14801},
      year={2025}
    }
