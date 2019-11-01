# Deep Crying: Recovering Blurred Images

## Data Extraction

### Datasets
- [Google Street View Dataset](https://www.crcv.ucf.edu/projects/GMCP_Geolocalization/#Dataset)
- MNIST
- CIFAR10

### Data Splitting

- Please refer to folder [data_preprocess](preprocessing)
- Used `SubsetRandomSampler` from `torch.utils.data.sampler` to split data
- Current distribution 7:2:1 (Train:Validation:Test)

### Data Visualization

- Please refer to folder [data_preprocess](preprocessing)


### Image blurring
 We will blur the images for model testing purpose by implement mosaic
 
#### Mosaic
- Mosaic the middle part of each photo.
- Each photo has been processed in four different degrees.

#### Small Examples

Clear Image

<img src="imgs/clear_demo1.png " alt="clear_demo" width="400"/>

Mosaic Image

<img src="imgs/mosaic_demo1.png " alt="clear_demo" width="400"/>

Small Sample Distribution

<img src="imgs/data_distribution_demo.png " alt="mosaic_demo" width="400"/>


