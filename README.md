# CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer

This is the code implementation of our paper.

[<video src="images/fig1.avi"></video>](https://github.com/JarrentWu1031/CCPL/issues/1#issue-1288394968)


### Requirements

This code is tested under Ubuntu 14.04 and 16.04. The total project can well function under the following environment: 

* python-3.6 
* pytorch >= 1.2
* torchvision >= 0.4
* tensorboardX >= 1.8
* other packages under python-3.6

### Preparations

Download [vgg_normalized.pth](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) and put them under `models/`.


### Train

To train a model, you need to first download content dataset and style dataset to the folder. We use COCO2014 as content dataset and wikiart as style dataset to train our model. Use command: `python train.py --content_dir <content_dir> --style_dir <style_dir> --log_dir <where to place logs> --save_dir <where to place the trained model> --training_mode <artistic or photo-realistic> `

### Test

To test a model, use commands like `python test.py --content input/content/lenna.jpg --style input/style/in2.jpg --decoder <decoder_dir> --SCT <SCT_dir> --testing_mode <artistic or photo-realistic>` for images and `python test_video_frame.py --content_dir <video frames dir> --style_path input/style/in2.jpg --decoder <decoder_dir> --SCT <SCT_dir> --testing_mode <artistic or photo-realistic> ` 

For more details and parameters, please refer to --help option.

### Pre-trained Models

To use the pre-trained models, please download here [pre-trained models](https://drive.google.com/drive/folders/1h6SZnZctkOv0b4dZVTZeONDAdlyg-A2f?usp=sharing) and specify them during training (These pre-trained models are trained under pytorch-1.9.1 and torchvision-0.10.1)

### Acknowledgments

The code is based on project [AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation). We sincerely thank them for their great work.
