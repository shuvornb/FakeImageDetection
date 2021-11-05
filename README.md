# Deep Fake Detection
Contains code for deep fake Detection

# Installation
```bash
git clone https://github.com/shuvornb/FakeImageDetection.git

pip install -r requirements.txt

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