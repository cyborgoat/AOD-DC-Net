# Deep Crying: Recovering Foggy/Hazy Images

## How to Run the code

```
$ python3 model_train.py
$ python3 run_dehazer.py
```

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

<img src="imgs/clear_demo1.jpg " alt="clear_demo" width="400"/>

Mosaic Image

<img src="imgs/foggy_demo1.jpg " alt="clear_demo" width="400"/>

First Run of Result

**Note**
  - Left side is the foggy image and right side is result image
  - The model only ran for 1 epoch from 6 images, so the current result might not be well performed and highly overexposure.
  
<img src="imgs/result_demo1.jpg " alt="clear_demo" width="400"/>


