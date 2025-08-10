# GenLV 
### Learning A Low-Level Vision Generalist via Visual Task Prompt [[Paper Link]](http://arxiv.org/abs/2408.08601)

Xiangyu Chen, Yihao Liu, Yuandong Pu, Wenlong Zhang, Jiantao Zhou, Yu Qiao and Chao Dong

### Quick Start
+ Environment
  + pytorch>=1.7
  + basicsr==1.4.2
+ Weights
  + Download the weights from [this link](https://huggingface.co/Kaiwen-Zhu/GenLV/tree/main/GenLV-30/ckpt) and put them in `conference_version/ckpt`.


### How to Inference
```sh
cd conference_version
python inference.py --model_size giant --input example/input.png --prompt_input example/prompt_input.png --prompt_target example/prompt_target.png --output_path example/prediction.png
```

### Results

### Citation

    @inproceedings{chen2024learning,
      title={Learning A Low-Level Vision Generalist via Visual Task Prompt},
      author={Xiangyu Chen and Yihao Liu and Yuandong Pu and Wenlong Zhang and Jiantao Zhou and Yu Qiao and Chao Dong},
      booktitle={ACM Multimedia 2024},
      year={2024}
    }