# UC_project
This repository contains the codebase for the project "ResNet: Predict Deviance of Los Angeles".

The [dataset](https://drive.google.com/file/d/1yRjrddwWMhIt73nSmd9RF01alXQjlToA/view?usp=sharing) contains the a set of sequential images which contain deviant places based on the open data provided by the Los Angeles Police Department and a set of data of Seoul provided by [Park et al.](https://deviance-project.github.io/DevianceNet/).

## File Structure
* Models
  * resnet.py - code for the 3D-ResNet model
  * resnet2p1.py - code for the (2+1)D-ResNet model
  * cnn.py - code for the plain CNN model
* LoadDataset.py - code for loading train and test data set
* main.py - the process of training, fine-tuning and testing model
* plot.py - make plots
