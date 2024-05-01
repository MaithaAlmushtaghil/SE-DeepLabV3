# SE-DeepLabV3
Ai702 Project\Group_11: SE-DeepLabV

## Description

Stenosis refers to the narrowing of a vessel due to the build-up of plaque, leading to the restriction in blood filled oxygen flow to the myocardium. This blockage leads to Coronary Artery Disease (CAD); one of the leading causes of death worldwide. Hence, we propose our project with a novel segmentation framework, SE-DeepLabV3, aimed at enhancing the localization and segmentation of Coronary Artery Stenosis regions, which is critical indicator of CAD severity. We leverage the DeepLabV3 architecture with using ResNet50 as a backbone, then we add squeeze-and-excitation (SE) blocks to the residual blocks resulting in a robust feature extractor. Furthermore, we employ homomorphic filtering to amplify vessel borders and stenosis regions. Our experimental results demonstrate a significant performance increase, with improvements in Dice Coefficient from 0.4561 to 0.4831, Intersection over Union from 0.8261 to 0.8356, and Pearson Correlation from 0.4743 to 0.4994, showcasing the effectiveness of our approach in accurately identifying stenotic regions, which is an initial step step in CAD diagnosis and monitoring.

## Files

### 1. [baselines_train.ipynb]
This jupyter notebook has the training of the baseline model we used, which is DeepLabV3 with Resnet50 backbone.

### 2. [DeepLabv3_SE.ipynb]
This jupyter notebook has the training of the model after adding the SE block to DeepLabV3 with Resnet50 backbone.

### 3. [filter_SE_resnet50.ipynb]
This jupyter notebook has the training of the model after adding the SE block and the homomorphic filter --> it's application was done in (preprocess.ipynb).

## Running Tests

To run the model on the testing set, follow these steps:

1. **Step 1:**: the dataset with the csv file of the splits are uploaded on the drive, please download them and adjust the paths in the code as needed.
2. **Step 2:**: in testing_models.ipynb run the cell corresponding to the model experiment, the weights for evaluation are uploaded on the drive, please download them and adjust the paths in the code as needed.
3. **Step 3:**: Run the cell of the model of choice, then run the evalation/testing cells.
