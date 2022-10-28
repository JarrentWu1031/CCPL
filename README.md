# CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer (ECCV 2022 **Oral**)

[**Paper**](https://arxiv.org/abs/2207.04808) | [**Video Demo**](https://youtu.be/scZuJCXhL14) | [**Web Demo**](https://replicate.com/jarrentwu1031/ccpl) | [**Supp File**](https://drive.google.com/file/d/1uXfWAD67gWTWUMSbeIjIONSFBKKw53BG/view?usp=sharing)

```
@inproceedings{wu2022ccpl,
  title={CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer},
  author={Wu, Zijie and Zhu, Zhen and Du, Junping and Bai, Xiang},
  booktitle={European Conference on Computer Vision},
  pages={189--206},
  year={2022},
  organization={Springer}
}
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/108389661/176405561-8a5153dc-6c70-4f34-9113-850bb4705266.gif" alt="animated" />
</p>


### Requirements

This code is tested under Ubuntu 14.04 and 16.04. The total project can well function under the following environment: 

* python-3.6 
* pytorch >= 1.2
* torchvision >= 0.4
* tensorboardX >= 1.8
* other packages under python-3.6

or simply run: 
```
pip install -r requirements.txt
```

### Inspirations for CCPL

<p align="center">
  <img src="https://github.com/JarrentWu1031/CCPL/blob/main/images/trim1.gif" alt="animated" width=80%/>
</p>

### Details of CCPL

<p align="center">
  <img src="https://github.com/JarrentWu1031/CCPL/blob/main/images/trim2.gif" alt="animated" width=80%/>
</p>

### Artistic Style Transfer

<div align=center>
<img src="https://github.com/JarrentWu1031/CCPL/blob/main/images/Fig.%206.jpg" width=85%>
</div>

### Photo-realistic Style Transfer

<div align=center>
<img src="https://github.com/JarrentWu1031/CCPL/blob/main/images/Fig.%207.jpg" width=85%>
</div>

### Super-resolution PST

<div align=center>
<img src="https://github.com/JarrentWu1031/CCPL/blob/main/images/pho1.jpg" width=85%>
</div>

### Short-term Temporal Consistency

<div align=center>
<img src="https://github.com/JarrentWu1031/CCPL/blob/main/images/Fig.%205.jpg" width=85%>
</div>

### Long-term Temporal Consistency

<div align=center>
<img src="https://github.com/JarrentWu1031/CCPL/blob/main/images/lterm.jpg" width=85%>
</div>

### Image-to-image translation

<p align="center">
  <img src="https://github.com/JarrentWu1031/CCPL/blob/main/images/fig10.gif" alt="animated" />
</p>

### Preparations

Download [vgg_normalized.pth](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) and put them under `models/`. Download [COCO2014 dataset](http://images.cocodataset.org/zips/train2014.zip) (content dataset) and [Wikiart dataset](https://www.kaggle.com/c/painter-by-numbers) (style dataset)
  
### Train

To train a model, use command like: 
<br>
```
python train.py --content_dir <content_dir> --style_dir <style_dir> --log_dir <where to place logs> --save_dir <where to place the trained model> --training_mode <artistic or photo-realistic> --gpu <specify a gpu>
```
or:
<br>
```
sh scripts/train.sh
```

### Test

To test a model, use commands like 
<br>
```
python test.py --content input/content/lenna.jpg --style input/style/in2.jpg --decoder <decoder_dir> --SCT <SCT_dir> --testing_mode <artistic or photo-realistic>
python test_video_frame.py --content_dir <video frames dir> --style_path input/style/in2.jpg --decoder <decoder_dir> --SCT <SCT_dir> --testing_mode <artistic or photo-realistic> 
```
or:
<br>
```
sh scripts/test.sh
sh scripts/test_video_frame.sh
```
To be noted, test_video_frame.py receives video frames as content inputs. 

For more details and parameters, please refer to --help option.

### Pre-trained Models

To use the pre-trained models, please download here [pre-trained models](https://drive.google.com/drive/folders/1XxhpzFqCVvboIyXKLfb2ocJZabPYu3pi?usp=sharing) and specify them during training (These pre-trained models are trained under pytorch-1.9.1 and torchvision-0.10.1)

### Acknowledgments

The code is based on project [AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation). We sincerely thank them for their great work.
