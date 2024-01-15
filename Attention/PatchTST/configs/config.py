config = {
    "files": {
        "data_submission": "../data/sample_submission.csv",
        "data_train": "../data/train.csv",
        "data_test":"../data/test.csv",
        "output": "./submit/model_",
        "submission":"./submit/submission_",
        "name": "BEST_D_Day"
    },
    "model_params": {
        "tst_size" : 96,
        "patch_size" : 16, 
        "n_patch" : 64,    #시간 단위?
        "hidden_dim": 256,
        "prediction_size": 96,
        "head_num":32,
        "layer_num":8
    },
    "train_params": {
        "data_loader_params": {
            "batch_size": 128,
            "shuffle": True,
        },
        "optim_params": {"lr": 0.0001},   #0.0000001
        "device": "cuda",
        "epochs": 100,
        "pbar": True,
        "min_delta": 0,
        "patience": 5,
        "early_stop": False
    },
    "train": True,
    "validation": False,
    "scheduler": True, 
    "nomal": True,
    "multi": False,
    "resnet": True
}
