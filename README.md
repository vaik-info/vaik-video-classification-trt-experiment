# vaik-video-classification-trt-experiment
Create json file by video classification model. Calc ACC.

## Install

```shell
pip install -r requirements.txt
```


## Docker Install

- amd64(g4dn.xlarge)

```shell
docker build -t g4dnxl_ed_experiment -f ./Dockerfile.g4dn.xlarge .
sudo docker run --runtime=nvidia \
           --name g4dnxl_ed_experiment_container \
           --rm \
           -v ~/.vaik-mnist-classification-dataset:/workspace/vaik-mnist-classification-dataset \
           -v ~/output_trt_model:/workspace/output_trt_model \
           -v $(pwd):/workspace/source \
           -it g4dnxl_ed_experiment /bin/bash
```

- arm64(JetsonXavierNX)

```shell
sudo docker build -t jxnj502_experiment -f ./Dockerfile.jetson_xavier_nx_jp_502 .
sudo docker run --runtime=nvidia \
           --name jxnj502_experiment_container \
           --rm \
           -v ~/.vaik-mnist-classification-dataset:/workspace/vaik-mnist-classification-dataset \
           -v ~/output_trt_model:/workspace/output_trt_model \
           -v $(pwd):/workspace/source \
           -it jxnj502_experiment /bin/bash
```

---------

## Install

```shell
pip install -r requirements.txt
```

---------
]
## Usage

### Create json file

```shell
python inference.py --skip_frame 16 \
                --input_saved_model_path '~/model.tflite' \
                --input_classes_path '~/.vaik-utc101-video-classification-dataset_tfrecords/train/ucf101_labels.txt' \
                --input_video_dir_path '~/.vaik-utc101-video-classification-dataset/test' \
                --output_json_dir_path '~/.vaik-video-classification-pb-experiment/test_inf'
```

- input_video_dir_path
    - example

```shell
.
.
├── test
│   ├── ApplyEyeMakeup
│   │   ├── ApplyEyeMakeup_1007.mp4
│   │   ├── ApplyEyeMakeup_1014.mp4
│   │   ├── ApplyEyeMakeup_1138.mp4
・・・
│   ├── ApplyLipstick
│   │   ├── ApplyLipstick_1161.mp4
│   │   ├── ApplyLipstick_1235.mp4
│   │   ├── ApplyLipstick_1342.mp4
・・・
```

#### Output
- output_json_dir_path
    - example

```json
{
  "answer": "ApplyEyeMakeup",
  "inf": [
    {
      "end_frame": 16,
      "label": [
        "ApplyLipstick",
        "BlowingCandles",
        "BoxingSpeedBag",
        "CuttingInKitchen",
        "BlowDryHair",
        "ApplyEyeMakeup",
        "BrushingTeeth",
        "Archery",
        "Bowling",
        "BabyCrawling",
        "BoxingPunchingBag",
        "Billiards",
        "CleanAndJerk",
        "BenchPress",
        "Biking",
        "Basketball",
        "CricketShot",
        "CliffDiving",
        "BodyWeightSquats",
        "BalanceBeam",
        "BasketballDunk",
        "BreastStroke",
        "BandMarching",
        "CricketBowling",
        "BaseballPitch"
      ],
      "score": [
        5.99818229675293,
        5.331709861755371,
        4.273311138153076,
        4.086155891418457,
        3.075073480606079,
        2.8466880321502686,
        2.111257314682007,
        1.962202548980713,
        1.843916893005371,
        1.5590394735336304,
        1.408070683479309,
        0.3319084644317627,
        -0.5513471961021423,
        -0.6263744235038757,
        -1.3993675708770752,
        -2.122079372406006,
        -2.80260968208313,
        -2.889033555984497,
        -3.2077112197875977,
        -3.549764394760132,
        -3.8684756755828857,
        -4.0212297439575195,
        -5.728559970855713,
        -5.763431072235107,
        -7.228459358215332
      ],
      "start_frame": 0
    },
・・・
  ],
  "video_path": "~/.vaik-utc101-video-classification-dataset/test/ApplyEyeMakeup/ApplyEyeMakeup_142.avi"
}
```
-----

### Calc ACC

```shell
python calc_topk_acc.py --top_k 3 \
                --input_json_dir_path '~/.vaik-video-classification-pb-experiment/test_inf' \
                --input_classes_path '~/.vaik-utc101-video-classification-dataset_tfrecords/train/sub_ucf101_labels.txt'
```

#### Output

``` text
               precision    recall  f1-score   support

ApplyEyeMakeup     0.8627    1.0000    0.9263        44
        Biking     0.7778    0.9211    0.8434        38
  CleanAndJerk     0.6750    0.8182    0.7397        33
  FrisbeeCatch     0.7955    0.9459    0.8642        37
     HorseRace     0.8500    0.9714    0.9067        35
      LongJump     0.8438    0.6923    0.7606        39
   PlayingDhol     0.8889    0.6531    0.7529        49
         Punch     1.0000    0.7692    0.8696        39
        Skiing     0.8000    0.9000    0.8471        40
        TaiChi     0.5263    0.3571    0.4255        28

      accuracy                         0.8115       382
     macro avg     0.8020    0.8028    0.7936       382
  weighted avg     0.8146    0.8115    0.8042       382
```
