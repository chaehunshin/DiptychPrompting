# Diptych Prompting (CVPR 2025)

<img src='./assets/teaser.png' width='100%' />
<br>

<a href="https://arxiv.org/abs/2411.15466"><img src="https://img.shields.io/badge/ariXv-2411.15466-A42C25.svg" alt="arXiv"></a>
<a href='https://diptychprompting.github.io'><img src='https://img.shields.io/badge/Project-Page-green'></a>


> **Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator**
> <br>
> Chaehun Shin,
> Jooyoung Choi,
> Heeseung Kim,
> Sungroh Yoon,
> Data Science and Artificial Intelligence Lab, Seoul National University
> <br>

---

## Introduction 
We introduce Diptych Prompting, a novel zero-shot approach that reinterprets as an inpainting task with precise subject alignment by leveraging the emergent property of diptych generation in large-scale text-to-image models. 
Diptych Prompting arranges an incomplete diptych with the reference image in the left panel, and performs text-conditioned inpainting on the right panel. 
We further prevent unwanted content leakage by removing the background in the reference image and improve fine-grained details in the generated subject by enhancing attention weights between the panels during inpainting. 

## How to Use

> [!NOTE]
> **Memory consumption**
>
> Flux can be quite expensive to run on consumer hardware devices and as a result this implementation comes with high memory requirements exceeding 40GB of VRAM .

### Setup (Optional)
1. **Environment setup**
```bash
conda create -n diptychprompting python=3.10
conda activate diptychprompting
```
2. **Requirements installation**
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Usage example
```bash
python diptych_prompting_inference.py --input_image_path /path/to/your-image --subject_name "name of your subject" --target_prompt "text prompt you want to generate"
```

For example, 
```bash
python diptych_prompting_inference.py --input_image_path ./assets/bear_plushie.jpg --subject_name "bear plushie" --target_prompt "a bear plushie riding a skateboard"
```
## License
Some of the implementations are based on [FLUX-Controlnet-Inpainting](https://github.com/alimama-creative/FLUX-Controlnet-Inpainting), and thus parts of the code would apply under the license of it.

## Citation
```
@article{shin2024largescaletexttoimagemodelinpainting,
      title={Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator}, 
      author={Chaehun Shin and Jooyoung Choi and Heeseung Kim and Sungroh Yoon},
      journal={arXiv preprint arXiv:2411.15466},
      year={2024}
}
```