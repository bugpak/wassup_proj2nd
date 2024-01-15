import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import glob
import argparse
from nn.model import PatchTST
from datasets.dataset import get_time_series
from metric.metric import metric
from metric.visualization import get_r2_graph, get_graph
from datasets.timeseries import PatchTSDataset


def predict(model:nn.Module, dl:torch.utils.data.DataLoader) -> np.array:
    with torch.inference_mode():
        for x in dl:
            x = x[0].to(device)
            out = model(x).detach().cpu().numpy()
        out = np.concatenate([out[:,0], out[-1,1:]])

    return out

def dynamic_predict(model:nn.Module, t_data:PatchTSDataset, params:dict) -> list:
    pred = []
    x,out = t_data[len(t_data)]
    model = model.cpu()
    input_size = params['prediction_size']
    with torch.inference_mode():
        for _ in range(int(params['tst_size']/params['prediction_size'])):
            x = np.concatenate([x,out],dtype=np.float32)[-input_size:]
            x = torch.tensor(x).cpu()
            x = x.reshape(1,-1)
            out = model(x).detach().cpu().numpy()
            pred.append(out)         
        pred = np.concatenate(pred)
        
    return pred

def main(args):
    train_params = args.get("train_params")
    files_ = args.get("files")
    device = torch.device(train_params.get("device"))
    params = args.get("model_params")
    dl_params = train_params.get("data_loader_params")

    df = pd.read_csv("../../../../../estsoft/data/train.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])

    ## make time-series dataset for input data
    df_pc, df_main = get_time_series(df)

    # real trarget data to make graph
    real = df_main[-params['tst_size']:].values
    nets = {
    "multi_channel":[],
    "nomal":[],
    "resnet":[]
    }
    
    if args.get("nomal"):
        nets['nomal'] = (PatchTST(params["n_patch"], params["patch_size"], params["hidden_dim"], 
                   params["head_num"], params["layer_num"], params["prediction_size"]).to(device))
    if args.get("multi"):
        nets['multi_channel'] = (MultitaskNN(params['input_size'],params['pred_size'],params['hidden_dim'],3).to(device))
    print(nets.values())
    for i,net in enumerate(nets):
        if torch.load(files_.get("output")+files_.get("name")+'_earlystop.pth'):
            nets[net].load_state_dict(torch.load(files_.get("output")+files_.get("name")+'_earlystop.pth'))
        else:
            nets[net].load_state_dict(torch.load(files_.get("output")+str(i)+files_.get("name")+'.pth'))

        nets[net].eval()
            
        # visualization model's performance
        out_dynamic = dynamic_predict(nets[net], tst, params)
        pred_dynamic = out_dynamic.squeeze(0)
        pred_dynamic = torch.tensor(pred_dynamic)
        val_score = metric(pred, real)
        pred_score = metric(pred_dynamic, real)
        print('val score : ',val_score)
        print('pred score : ',pred_score)
        get_r2_graph(pred_dynamic, real, pred, real, str(i)+'_'+files_.get("name"))

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
    parser.add_argument(
        "-c", "--config", default="./configs/config.py", type=str, help="configuration file"
    )
    parser.add_argument(
        "-mode", "--multi-mode", default=False, type=bool, help="multi train mode"
    )

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(args.config)
    exec(open(args.config).read())
    main(config)
    
    if args.multi_mode:
        for filename in glob.glob("config/multi*.py"):
            print(filename)
            exec(open('./'+filename).read())
            main(config)