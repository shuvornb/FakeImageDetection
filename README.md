# Deep Fake Detection
Contains code for deep fake Detection

# Installation
```bash
git clone https://github.com/shuvornb/FakeImageDetection.git

To install required library-
pip install -r requirements.txt
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


To extract face frames from a video-
python3 video_to_face_extractor.py --input_path=fakeimagedetection/sample_data/sample_video/from --output_path=fakeimagedetection/sample_data/sample_video/to/ --image_shape=64

To shuffle and shift images from one folder to another:
python3 shuffle_shift_data.py --input_path=fakeimagedetection/sample_data/sample_shift/from/ --output_path=fakeimagedetection/sample_data/sample_shift/to/ --shift_amount=3

To train a model on data_256:
1) Meso4: 

python3 driver.py --mode=train --model_name=meso4 --data_path=fakeimagedetection/sample_data/data_256/df

2) Xception:

python3 driver.py --mode=train --model_name=xception --data_path=fakeimagedetection/sample_data/data_256/df

To test a model on data_256
python3 driver.py --mode=test --test_model_path=fakeimagedetection/saved_models/meso4_df_tr_ts_Tue_Nov_30_15_51_21_2021/ --data_path=fakeimagedetection/sample_data/data_256/df


To test a model on data_64
python3 driver.py --mode=test --test_model_path=fakeimagedetection/saved_models/meso4_df_tr_ts_Tue_Nov_30_15_51_21_2021/ --data_path=fakeimagedetection/sample_data/data_64/df

To test a model on data_64 with EDSR Upsampling
python3 driver.py --mode=test --test_model_path=fakeimagedetection/saved_models/meso4_df_tr_ts_Tue_Nov_30_15_51_21_2021/ --data_path=fakeimagedetection/sample_data/data_64/df --upsample=True






python3  driver.py --help 
```
Example:
```bash
python3 driver.py --mode=train --model_name=meso4 --data_path=fakeimagedetection/sample_data/deepfake
```

Trained model will be saved inside fakeimagedetection/saved_models/ and training results will be saved inside fakeimagedetection/results

```bash
python3 driver.py --mode=test --test_model_path=fakeimagedetection/saved_models/meso4 --data_path=fakeimagedetection/sample_data/deepfake
```