# An End-to-end Image Dehazing System with Light-weighted CNN And Dark Channel Prior

## Contributors

Instructor: Joseph Konan

Team: Jingsi Chen,Junxiao Guo, Shuhao Ren ,Luo Yu (ordered by last name initials)

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
- Please refer to folder [haze_generator](haze_generator)
 
#### Baseline Model Result

Clear Image

<img src="imgs/5_epoches_clear.jpg " alt="clear_demo" width="1000" height="200"/>

Hazy Image

<img src="imgs/5_epoches_hazy.jpg " alt="hazy_demo" width="1000" height="200"/>

Baseline Model Result

**Note**
  - The model was trained with 5 epoch from 2300 images.
  
<img src="imgs/5_epoches_clean.jpg " alt="clean_demo" width="1000" height="200"/>


