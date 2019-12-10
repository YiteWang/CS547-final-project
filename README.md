# CS547 final project

* Team members: Yite Wang, Yuchen He, Jing Wu, Randy Chase

### How to use the Code

Put the dataset you want to folder `datasets`.

Notice the structure under datasets is as follows:

```
vangogh2photo
│
└───TrainA
│   │   
│   └───Apple_train
│       │   pic1.png
│       │   pic2.png
│       │   ...
│    
│   
└───TrainB
│   │   
│   └───Orange_train
│       │   pic1.png
│       │   pic2.png
│       │   ...
│    
│   
└───TestA
│   │   
│   └───Apple_test
│       │   pic1.png
│       │   pic2.png
│       │   ...
│    
│   
└───TestB
    │   
    └───Orange_test
        │   pic1.png
        │   pic2.png
        │   ...
```

Then run the following code in terminal:

`python main.py --epochs 200 --decay_epoch 100 --batch_size 2 --training True --testing True --data_name apple2orange`

If you want to do Monet, you should add identity loss, which needs extra arguments: `--use_id_loss True`

### Reference:

1.[Original CycleGAN paper](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

2.[Simple implementation of CycleGAN](https://github.com/arnab39/cycleGAN-PyTorch)
