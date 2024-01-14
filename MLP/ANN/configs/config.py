config = {
    "files": {
        "data_submission": "../data/sample_submission.csv",
        "data_train": "../data/train.csv",
        "data_test":"../data/test.csv",
        "output": "./submit/model_",
        "submission":"./submit/submission_",
        "name": "best_no_3"
    },
    "model_params": {
        "use_dropout": False,
        "pred_size":96,
        "tst_size":96,   #default
        "input_size":25,
        "hidden_dim":256,
        "channel_num":3
    },
    "train_params": {
        "data_loader_params": {
            "batch_size": 128,
            "shuffle": True,
        },
        "optim_params": {"lr": 0.0001},   #0.0000001
        "device": "cuda",
        "epochs": 5000,
        "pbar": True,
        "min_delta": 0,
        "patience": 5,
        "early_stop": False
    },
    "train": True,
    "validation": False,
    "scheduler": True, 
    "nomal": True,
    "multi": True,
    "resnet": True
}
