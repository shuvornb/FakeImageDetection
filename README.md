# Deep Fake Detection
The rapid progress in synthetic image generation and manipulation has now come to a point where it raises significant concerns on the social implications. This not only leads to a loss of trust in digital content, but also causes further harm by spreading false information. We are confronted today with manipulated visual content in almost any digital medium. Images and videos particularly of human faces are also victim of these manipulations. We have seen such manipulations to alter political leader’s speech jeopardizing the reputation of a nation. We have seen such manipulations being used to make revenge porn in last few years. With the development of more and more advanced techniques, fake contents are getting closer to real ones creating a severe threat. So, detection of these forgeries is of paramount importance. A lot of researchers have already addressed the problem from different perspectives. We explored the problem from two aspects. (a) Performance of fake detection models on generalizability (b) Performance of fake detection models on low resolution data. We conduct experiments to confirm the first observation. For the second one, we provide an improved model which yields a maximum of 34\% improvement in the test accuracy.

# Installation
```bash
git clone https://github.com/shuvornb/FakeImageDetection.git
```
To install required library-
```bash
pip install -r requirements.txt
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

# How to run the experiments:

To extract face frames from a video-
```bash
python3 video_to_face_extractor.py --input_path=fakeimagedetection/sample_data/sample_video/from --output_path=fakeimagedetection/sample_data/sample_video/to/ --image_shape=64
```
To shuffle and shift images from one folder to another:
```bash
python3 shuffle_shift_data.py --input_path=fakeimagedetection/sample_data/sample_shift/from/ --output_path=fakeimagedetection/sample_data/sample_shift/to/ --shift_amount=3
```
To train a model on data_256:
1) Meso4: 
```bash
python3 driver.py --mode=train --model_name=meso4 --data_path=fakeimagedetection/sample_data/data_256/df
```
2) Xception:
```bash
python3 driver.py --mode=train --model_name=xception --data_path=fakeimagedetection/sample_data/data_256/df
```
To test a model on data_256
```bash
python3 driver.py --mode=test --test_model_path=fakeimagedetection/saved_models/meso4_df_tr_ts_Tue_Nov_30_15_51_21_2021/ --data_path=fakeimagedetection/sample_data/data_256/df
```

To test a model on data_64
```bash
python3 driver.py --mode=test --test_model_path=fakeimagedetection/saved_models/meso4_df_tr_ts_Tue_Nov_30_15_51_21_2021/ --data_path=fakeimagedetection/sample_data/data_64/df
```
To test a model on data_64 with EDSR Upsampling
```bash
python3 driver.py --mode=test --test_model_path=fakeimagedetection/saved_models/meso4_df_tr_ts_Tue_Nov_30_15_51_21_2021/ --data_path=fakeimagedetection/sample_data/data_64/df --upsample=True
```
