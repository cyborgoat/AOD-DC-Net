# Deep Crying: Recovering Foggy/Hazy Images

## Data Extraction

### Datasets
- [Google Street View Dataset](https://www.crcv.ucf.edu/projects/GMCP_Geolocalization/#Dataset)
- Image Structure: 1281*1025

### Data Splitting

- Please refer to folder [data_preprocess](preprocessing)
- Used `SubsetRandomSampler` from `torch.utils.data.sampler` to split data
- Current distribution 8:2 (Train:Validation)

### Data Visualization

- Please refer to folder [data_preprocess](preprocessing)


### Create Foggy Image
 We will artificially make the images be foggy to train our model and compare the output result.
 
#### Small Examples

Clear Image

<img src="imgs/clear_demo1.png " alt="clear_demo" width="400"/>

Mosaic Image

<img src="imgs/mosaic_demo1.png " alt="clear_demo" width="400"/>

Small Sample Distribution

<img src="imgs/data_distribution_demo.png " alt="mosaic_demo" width="400"/>


