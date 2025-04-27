#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:09:30 2022

@author: boli
"""
# from mmcv import Config # Old import
from mmengine import Config # Correct import for mmengine
import pandas as pd
import os
import re # Needed for replace_cfg_vals
import prettytable # Needed for print_metrics
# Line length fix
from collections import OrderedDict # Unused?
import torch as th
import sys
import inspect
# Need Registry for build_from_cfg type hint, but it's not directly used here
# from mmengine.registry import Registry # We import it in builder.py
import plotly.graph_objects as go
import os.path as osp
import pickle # Needed for create_radar_score_baseline
from scipy.stats import norm # Needed for calculate_radar_score
import random # Needed for set_seed
import numpy as np # Needed for set_seed and evaluate_metrics
import matplotlib.pyplot as plt # Needed for plot

sys.path.append(".") # Not ideal, check if necessary

# Restore necessary functions

def set_seed(random_seed):
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)

def get_attr(args, key=None, default_value=None):
    if isinstance(args, dict):
        return args[key] if key in args else default_value
    elif isinstance(args, object):
        return getattr(args, key, default_value) if key is not None else default_value

def print_metrics(stats):
    table = prettytable.PrettyTable()
    # table.add_row(['' for _ in range(len(stats))])
    for key, value in stats.items():
        table.add_column(key, value)
    return table

def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    # Don't check registry type strictly here, rely on caller (builder.py)
    # if not isinstance(registry, Registry):
    #     raise TypeError('registry must be an mmengine.Registry object, '
    #                     f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')

def replace_cfg_vals(ori_cfg):
    """Replace the string "${key}" with the corresponding value.

    Replace the "${key}" with the value of ori_cfg.key in the config. And
    support replacing the chained ${key}. Such as, replace "${key0.key1}"
    with the value of cfg.key0.key1. Code is modified from `vars.py
    < https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/vars.py>`_  # noqa: E501

    Args:
        ori_cfg (mmengine.Config): # Adjusted type hint for mmengine
            The origin config with "${key}" generated from a file.

    Returns:
        updated_cfg [mmengine.Config]: # Adjusted type hint for mmengine
            The config with "${key}" replaced by the corresponding value.
    """

    def get_value(cfg, key):
        for k in key.split('.'):
            cfg = cfg[k]
        return cfg

    def replace_value(cfg):
        if isinstance(cfg, dict):
            return {key: replace_value(value) for key, value in cfg.items()}
        elif isinstance(cfg, list):
            return [replace_value(item) for item in cfg]
        elif isinstance(cfg, tuple):
            return tuple([replace_value(item) for item in cfg])
        elif isinstance(cfg, str):
            # the format of string cfg may be:
            # 1) "${key}", which will be replaced with cfg.key directly
            # 2) "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx",
            # which will be replaced with the string of the cfg.key
            keys = pattern_key.findall(cfg)
            values = [get_value(ori_cfg, key[2:-1]) for key in keys]
            if len(keys) == 1 and keys[0] == cfg:
                # the format of string cfg is "${key}"
                cfg = values[0]
            else:
                for key, value in zip(keys, values):
                    # the format of string cfg is
                    # "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx"
                    assert not isinstance(value, (dict, list, tuple)), \
                        f'for the format of string cfg is ' \
                        f"'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', " \
                        f"the type of the value of '${key}' " \
                        f'can not be dict, list, or tuple' \
                        f'but you input {type(value)} in {cfg}'
                    cfg = cfg.replace(key, str(value))
            return cfg
        else:
            return cfg

    # the pattern of string "${key}"
    pattern_key = re.compile(r'\$\{[a-zA-Z\d_.]*\}')
    # the type of ori_cfg._cfg_dict is mmengine.ConfigDict
    # Use ConfigDict directly as _cfg_dict might be internal
    updated_cfg = Config(
        replace_value(ori_cfg), filename=ori_cfg.filename)
    # replace the model with model_wrapper
    if updated_cfg.get('model_wrapper', None) is not None:
        updated_cfg.model = updated_cfg.model_wrapper
        updated_cfg.pop('model_wrapper')
    return updated_cfg

def plot(df,alg,color='darkcyan',save=False):
    x = range(len(df))
    if df.empty or "total assets" not in df.columns or len(df["total assets"].values) == 0:
         print("Warning: DataFrame is empty or missing 'total assets', cannot plot.")
         return
         
    start_asset = df["total assets"].values[0]
    if start_asset == 0:
         print("Warning: Initial asset value is zero, cannot calculate percentage return.")
         # Optionally plot raw values instead
         # y = df["total assets"].values
         # plt.ylabel('Total Assets',size=18)
         return
    else:
        y=(df["total assets"].values-start_asset)/start_asset
        plt.ylabel('Total Return(%)',size=18)
        
    plt.plot(x, y*100, color, label=alg)
    plt.xlabel('Trading times',size=18)
    plt.grid(ls='--')
    plt.legend(loc='upper center', fancybox=True, ncol=1, fontsize='x-large',bbox_to_anchor=(0.49, 1.15,0,0))
    if save:
        try:
            plt.savefig("{}.pdf".format(alg))
            print(f"Saved plot to {alg}.pdf")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close()
    else:
        plt.show()

def plot_metric_against_baseline(total_asset,buy_and_hold,alg,task,color='darkcyan',save_dir=None,metric_name='Total asset'):
    # print('total_asset shape is:',total_asset.shape)
    # print(total_asset)

    #normalize total_asset and buy_and_hold by the first value
    # print('total_asset shape is:',total_asset.shape,total_asset)
    if len(total_asset) == 0:
         print("Warning: total_asset is empty, cannot plot.")
         return
    
    # Ensure total_asset is a numpy array for safe division
    total_asset = np.array(total_asset)
    
    if total_asset[0] == 0:
        print("Warning: Initial total_asset is zero, cannot normalize. Plotting raw values.")
        normalized_asset = total_asset
        normalized_buy_and_hold = np.array(buy_and_hold) if buy_and_hold is not None else None
    else:
        normalized_asset = total_asset / total_asset[0]
        if buy_and_hold is not None:
            normalized_buy_and_hold = np.array(buy_and_hold) / total_asset[0]
        else:
            normalized_buy_and_hold = None

    x = range(len(normalized_asset))
    # print('total_asset shape is:',total_asset.shape)
    # print('x shape is:',len(x))
    # set figure size
    plt.figure(figsize=(10, 6))
    y=normalized_asset
    plt.plot(x, y, color, label=alg)
    plt.xlabel('Trading times',size=18)
    plt.ylabel(metric_name,size=18)
    if normalized_buy_and_hold is not None:
        # print('buy and hold shape is:',normalized_buy_and_hold.shape)
        plt.plot(x, normalized_buy_and_hold, 'r', label='Buy and Hold')
    plt.grid(ls='--')
    plt.legend(fancybox=True, ncol=1)
    # set title
    plt.title(f'{metric_name} of {alg} in {task}')
    if save_dir is not None:
        plt.savefig(osp.join(save_dir,f"Visualization_{task}.png"))
        plt.close() # Close the plot after saving to avoid displaying it if not needed
    else:
        plt.show()

def plot_radar_chart(data,plot_name,radar_save_path):
    data_list_profit=[]
    data_list_risk=[]
    # Use keys directly present in test_metrics_scores_dict based on calculate_radar_score
    for metric in ['Excess_Profit','sharpe_ratio','cr','sor']:
        data_list_profit.append(data[metric]+100)
    for metric in ['vol','mdd']:
        data_list_risk.append(data[metric]+100)
    Risk_Control=sum(data_list_risk)/len(data_list_risk)
    Profitability=sum(data_list_profit)/len(data_list_profit)
    fig = go.Figure()
    r_values = data_list_profit + data_list_risk
    theta_values = [0, 60, 120, 180, 240, 300]

    # Duplicate the first point at the end
    r_values.append(r_values[0])
    theta_values.append(theta_values[0])

    fig.add_trace(go.Scatterpolar(
        r=r_values,
        theta=theta_values,
        fill=None,
        line_color='peru',
        name='Metrics Radar'
    ))
    # print(data_list_profit+data_list_risk,Risk_Control,Profitability)
    fig.add_trace(go.Barpolar(
    r=[Profitability],
    theta=[90],
    width=[180],
    marker_color=["#E4FF87"],
    marker_line_color="black",
    marker_line_width=0.5,
    opacity=0.7,
    name='Profitability'
))
    fig.add_trace(go.Barpolar(
        r=[Risk_Control],
        theta=[270],
        width=[60],
        marker_color=['#709BFF'],
        marker_line_color="black",
        marker_line_width=0.5,
        opacity=0.7,
        name='Risk_Control'
    ))

    fig.update_layout(
        font_size=16,
        legend_font_size=22,
        template=None,
        barmode='overlay',
        polar=dict(
            radialaxis=dict(range=[0,200],visible=True, showticklabels=True, ticks=''
    ,tickvals = [0,50,100,150,200],
            ticktext = [-100,-50,0,50,100]
    ),
            angularaxis=dict(showticklabels=True, ticks='',
            tickvals=[0,60,120,180,240,300],
            ticktext=['Excess Profit', 'Sharp Ratio',
               'Calmar Ratio','Sortino Ratio']+['Volatility', 'Max Drawdown'])
        )
    )
    # ax = fig.add_subplot(111, polar=True)
    # ax.set_xticklabels([\'-100\',\'-50\',\'0\',\'50\',\'100\'])
    radar_save_name=osp.join(radar_save_path,plot_name).replace("\\", "/")
    fig.write_image(radar_save_name)
    # print('Radar plot printed to:', radar_save_name)
    return 0

def evaluate_metrics(scores_dicts,print_info=False):
    ##TODO: high frequency have different normalization factor
    time_scale_factor=252

    Excess_Profit_list = []
    daily_return_list = []
    tr_list = []
    mdd_list = []
    cr_list = []
    sor_list = [] # Added for Sortino calculation consistency
    total_assets_list = [] # Collect total assets for consistency

    for scores_dict in scores_dicts:
        # Ensure keys exist before accessing
        excess_profit = scores_dict.get('Excess Profit', 0)
        daily_return = scores_dict.get('daily_return', np.array([0]))
        total_assets = scores_dict.get('total_assets', np.array([1, 1])) # Default to avoid div by zero
        
        if len(total_assets) < 2:
             total_assets = np.array([1, 1]) # Ensure at least two points for return calc
        if len(daily_return) == 0:
             daily_return = np.array([0]) # Ensure daily_return is not empty

        Excess_Profit_list.append(excess_profit)
        total_assets_list.append(total_assets)
        daily_return_list.append(daily_return)

        # Calculate Total Return (tr)
        start_asset = total_assets[0] if total_assets[0] > 1e-10 else 1e-10
        end_asset = total_assets[-1]
        tr = (end_asset / start_asset) - 1
        tr_list.append(tr)

        # Calculate Max Drawdown (mdd)
        if len(total_assets) > 0:
            peak = np.maximum.accumulate(total_assets)
            drawdown = (peak - total_assets) / peak
            mdd = np.max(drawdown) if len(drawdown) > 0 else 0
        else:
            mdd = 0
        mdd_list.append(mdd)

        # Calculate Calmar Ratio (cr)
        cr = tr / (mdd + 1e-10)
        cr_list.append(cr)
        
        # Calculate Sortino Ratio (sor)
        neg_ret_lst = daily_return[daily_return < 0]
        downside_std = np.std(neg_ret_lst) if not neg_ret_lst.empty else 0
        # Using annualized mean return for consistency with Sharpe
        mean_return_annualized = np.mean(daily_return) * time_scale_factor 
        sor = mean_return_annualized / (downside_std * np.sqrt(time_scale_factor) + 1e-10)
        sor_list.append(sor)

    output_dict={}
    output_dict['Excess_Profit'] = np.mean(Excess_Profit_list) if Excess_Profit_list else 0
    output_dict['tr'] = np.mean(tr_list) if tr_list else 0
    
    daily_return_merged = np.concatenate(daily_return_list) if daily_return_list else np.array([0])
    
    # Calculate Sharpe Ratio
    mean_return_annualized = np.mean(daily_return_merged) * time_scale_factor
    std_dev_annualized = np.std(daily_return_merged) * np.sqrt(time_scale_factor)
    output_dict['sharpe_ratio'] = mean_return_annualized / (std_dev_annualized + 1e-10)
    
    # Calculate Volatility (annualized standard deviation)
    output_dict['vol'] = std_dev_annualized

    output_dict['mdd'] = np.mean(mdd_list) if mdd_list else 0
    output_dict['cr'] = np.mean(cr_list) if cr_list else 0
    output_dict['sor'] = np.mean(sor_list) if sor_list else 0 # Use mean of calculated Sortinos

    if print_info:
        # Use OrderedDict if needed, otherwise regular dict is fine in modern Python
        stats = {
            "Excess Profit": [f"{output_dict['Excess_Profit']:.4f}%"],
            "Total Return": [f"{output_dict['tr']:.4f}%"], # Added TR to printed stats
            "Sharp Ratio": [f"{output_dict['sharpe_ratio']:.4f}"],
            "Volatility": [f"{output_dict['vol']:.4f}"], # Annualized Vol
            "Max Drawdown": [f"{output_dict['mdd']:.4f}"],
            "Calmar Ratio": [f"{output_dict['cr']:.4f}"],
            "Sortino Ratio": [f"{output_dict['sor']:.4f}"]
        }
        print(f"--- {print_info} --- ")
        table = print_metrics(stats)
        print(table)
    return output_dict

def create_radar_score_baseline(dir_name,metric_path,zero_score_id='Do_Nothing',fifty_score_id='Blind_Bid'):
    # get 0-score metrics
    # noted that for Mdd and Volatility, the lower, the better.
    # So the 0-score metric for Mdd and Volatility here is actually 100-score

    # We assume that the score of all policy range within  (-100,100)
    # Do Nonthing policy will score 0
    # the baseline policy(Blind Buy for now) should score 50(-50 if worse than Do Nothing)
    # The distribution of the score of policies is a normal distribution
    # The Do Nothing policy is 0.5 percentile and baseline policy should be the 0.75 percentile(0.675 sigma away from Do Nothing)
    # Then we can score policies based on the conversion of sigma and metric value
    metric_path_zero=metric_path + '_'+zero_score_id
    zero_scores_files = [osp.join(dir_name,filename) for filename in os.listdir(dir_name) if filename.startswith(metric_path_zero)]
    zero_scores_dicts =[]
    for file in zero_scores_files:
        try:
            with open(file, 'rb') as f:
                zero_scores_dicts.append(pickle.load(f))
        except Exception as e:
            print(f"Error loading zero score file {file}: {e}")
            
    # get 50-score metrics
    metric_path_fifty=metric_path + '_'+fifty_score_id
    fifty_scores_files = [osp.join(dir_name,filename) for filename in os.listdir(dir_name) if filename.startswith(metric_path_fifty)]
    fifty_scores_dicts =[]
    for file in fifty_scores_files:
        try:
            with open(file, 'rb') as f:
                fifty_scores_dicts.append(pickle.load(f))
        except Exception as e:
            print(f"Error loading fifty score file {file}: {e}")
            
    # We only assume the daily return follows normal distribution so to give a overall metric across multiple tests we will calculate the metrics here.
    zero_metrics=evaluate_metrics(zero_scores_dicts,print_info=zero_score_id+' policy performance summary')
    fifty_metrics=evaluate_metrics(fifty_scores_dicts,print_info=fifty_score_id+' policy performance summary')

    metrics_sigma_dict={}
    # Calculate sigma based on difference between 50-score and 0-score (divided by 0.675 sigma difference)
    # Handle potential division by zero if metrics are identical or sigma calculation fails
    sigma_divisor = 0.675
    vol_mdd_sigma_divisor = (3 - 0.675) # Assuming 0-score for vol/mdd represents 3 sigma (best)

    def safe_sigma_calc(metric_name, divisor):
         diff = abs(fifty_metrics.get(metric_name, 0) - zero_metrics.get(metric_name, 0))
         return diff / divisor if divisor != 0 else 1e-9 # Return small value if divisor is zero

    metrics_sigma_dict['Excess_Profit'] = safe_sigma_calc('Excess_Profit', sigma_divisor)
    metrics_sigma_dict['tr'] = safe_sigma_calc('tr', sigma_divisor)
    metrics_sigma_dict['sharpe_ratio'] = safe_sigma_calc('sharpe_ratio', sigma_divisor)
    metrics_sigma_dict['cr'] = safe_sigma_calc('cr', sigma_divisor)
    metrics_sigma_dict['sor'] = safe_sigma_calc('sor', sigma_divisor)
    # vol and mdd: zero_metrics represents the best score (lower is better), assumed at 3 sigma
    metrics_sigma_dict['vol'] = safe_sigma_calc('vol', vol_mdd_sigma_divisor) 
    metrics_sigma_dict['mdd'] = safe_sigma_calc('mdd', vol_mdd_sigma_divisor)
    
    # Ensure sigma is not zero to prevent division by zero later
    for k, v in metrics_sigma_dict.items():
        if v == 0:
            print(f"Warning: Calculated sigma for {k} is zero. Setting to small value.")
            metrics_sigma_dict[k] = 1e-9
            
    return metrics_sigma_dict, zero_metrics

def calculate_radar_score(dir_name,metric_path,agent_id,metrics_sigma_dict,zero_metrics):
    metric_path = metric_path + '_'+agent_id
    # print(metric_path)
    # print(os.listdir(dir_name))
    test_scores_files = [osp.join(dir_name,filename) for filename in os.listdir(dir_name) if filename.startswith(metric_path)]
    test_scores_dicts = []
    for file in test_scores_files:
        try:
            with open(file, 'rb') as f:
                test_scores_dicts.append(pickle.load(f))
        except Exception as e:
            print(f"Error loading test score file {file}: {e}")
            
    if not test_scores_dicts: # Handle case where no score files were found/loaded
         print(f"Warning: No score files found or loaded for agent {agent_id}. Returning zero scores.")
         # Return a dictionary with zero scores for all expected metrics
         zero_scores = {
            'Excess_Profit': 0.0, 'tr': 0.0, 'sharpe_ratio': 0.0, 'cr': 0.0, 'sor': 0.0,
            'vol': 0.0, 'mdd': 0.0, 'Profitability': 0.0, 'Risk Control': 0.0
         }
         return zero_scores
         
    test_metrics=evaluate_metrics(test_scores_dicts,print_info='Tested '+agent_id+' policy performance summary')
    #turn metrics to sigma
    profit_metric_names=['Excess_Profit','tr','sharpe_ratio','cr','sor']
    risk_metric_names = ['vol', 'mdd']
    test_metrics_scores_dict={}
    for metric_name in profit_metric_names:
        sigma = metrics_sigma_dict.get(metric_name, 1e-9) # Use stored sigma, default to small value
        test_metrics_scores_dict[metric_name]=norm.cdf((test_metrics.get(metric_name, 0)-zero_metrics.get(metric_name, 0))/sigma)*200-100
    for metric_name in risk_metric_names:
        sigma = metrics_sigma_dict.get(metric_name, 1e-9)
        # Higher sigma (better risk score) corresponds to lower metric value (vol/mdd)
        # Score = CDF( (Target Sigma - Calculated Sigma) )
        # Target Sigma for vol/mdd is 3 (best score)
        # Calculated Sigma for vol/mdd = (Metric Value - Zero Metric) / Sigma_per_Metric_Unit
        # We calculate the score based on how far the metric is from the zero (best) metric, scaled by sigma
        # Score should decrease as metric increases (worsens)
        # Using norm.cdf(3 - (deviation / sigma)) scales it correctly: 0 deviation -> cdf(3) -> ~1.0 -> score 100
        deviation = test_metrics.get(metric_name, 0) - zero_metrics.get(metric_name, 0)
        scaled_deviation_sigma = deviation / sigma
        # Score = CDF(BestPossibleSigma - DeviationInSigmaUnits) * 200 - 100
        test_metrics_scores_dict[metric_name] = norm.cdf(3 - scaled_deviation_sigma) * 200 - 100
        
    # Aggregate scores
    profit_scores = [test_metrics_scores_dict.get(m, 0) for m in ['tr','sharpe_ratio','cr','sor']] # Exclude Excess Profit?
    test_metrics_scores_dict["Profitability"] = np.mean(profit_scores) if profit_scores else 0
    
    risk_scores = [test_metrics_scores_dict.get(m, 0) for m in risk_metric_names]
    test_metrics_scores_dict["Risk Control"] = np.mean(risk_scores) if risk_scores else 0

    # Prepare dict for printing
    test_metrics_scores_dict_for_print = {
        "Excess Profit": [f"{test_metrics_scores_dict.get('Excess_Profit', 0):.2f}"],
        "Sharp Ratio": [f"{test_metrics_scores_dict.get('sharpe_ratio', 0):.2f}"],
        "Volatility": [f"{test_metrics_scores_dict.get('vol', 0):.2f}"],
        "Max Drawdown": [f"{test_metrics_scores_dict.get('mdd', 0):.2f}"],
        "Calmar Ratio": [f"{test_metrics_scores_dict.get('cr', 0):.2f}"],
        "Sortino Ratio": [f"{test_metrics_scores_dict.get('sor', 0):.2f}"]
    }
    print('Tested scores are:')
    print(print_metrics(test_metrics_scores_dict_for_print))
    return test_metrics_scores_dict

def MRL_F2B_args_converter(args):
    output_args={}
    output_args['data_path']=args['dataset_path']
    output_args['method']='slice_and_merge'
    if args['dataset_name']=="order_excecution:BTC":
        output_args['OE_BTC']=True
    else:
        output_args['OE_BTC']=False

    if args['labeling_method']=="slope":
        # use auto zooming for slope
        output_args['slope_interval']=[0,0]
    # keep the same for the rest of parameters in the args for output_args
    for key in args.keys():
        if key not in output_args.keys():
            output_args[key]=args[key]
    return output_args

# def get_mapper(name: str):
#     if name == "general_mapping":

def plot_log_trading_decision_on_market(market_features_dict, trading_points, alg, task, color='darkcyan', save_dir=None, metric_name='Level 0 Bid and Ask Distance'):
    # parse market_features_dict to get market_features
    market_features=list(market_features_dict.keys())
    if not market_features:
         print("Warning: market_features_dict is empty, cannot plot.")
         return
         
    x = range(len(market_features_dict[market_features[0]]))
    # create a pd.DataFrame to store trading logs of x rows
    trading_log = pd.DataFrame(index=x)
    # print('total_asset shape is:',total_asset.shape)
    # print('x shape is:',len(x))
    # set figure size
    fig, ax1 = plt.subplots(figsize=(20, 6)) # Create figure and primary axes

    # Plot market features on ax1
    color_cycle = plt.get_cmap("tab10")
    for i, market_feature in enumerate(market_features):
        trading_log[market_feature]=market_features_dict[market_feature]
        y=market_features_dict[market_feature]
        ax1.plot(x, y, label=market_feature, color=color_cycle(i), drawstyle='steps-post')
    ax1.set_xlabel('Trading times',size=12)
    ax1.set_ylabel('Market Features', size=12)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.tick_params(axis='y')

    # Create secondary axis for trading decisions
    ax2 = ax1.twinx()
    buy_trade_points=trading_points.get('buy', {}) # Use .get for safety
    sell_trade_points=trading_points.get('sell', {})

    # Combine points for scaling
    all_volumes = list(buy_trade_points.values()) + list(sell_trade_points.values())
    scale = max(all_volumes) if all_volumes else 1 # Avoid max of empty sequence

    # Prepare bar data
    buy_indices = list(buy_trade_points.keys())
    buy_volumes = list(buy_trade_points.values())
    sell_indices = list(sell_trade_points.keys())
    sell_volumes = [-v for v in sell_trade_points.values()] # Sell volumes as negative

    # Plot bars on ax2
    bar_width = 0.8
    if buy_indices:
        ax2.bar(buy_indices, buy_volumes, width=bar_width, label='Buy', color='red', alpha=0.7)
        # Log buy trades
        for idx, vol in buy_trade_points.items():
            if idx in trading_log.index:
                 trading_log.loc[idx,'buy'] = vol
            else:
                 print(f"Warning: Buy index {idx} out of bounds for logging.")
                 
    if sell_indices:
        ax2.bar(sell_indices, sell_volumes, width=bar_width, label='Sell', color='green', alpha=0.7)
        # Log sell trades (log positive volume)
        for idx, vol in sell_trade_points.items():
             if idx in trading_log.index:
                  trading_log.loc[idx,'sell'] = vol
             else:
                  print(f"Warning: Sell index {idx} out of bounds for logging.")

    # Configure ax2
    ax2.set_ylabel('Trading Action (Volume)', size=12)
    ax2.set_ylim(-1.2 * scale, 1.2 * scale)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.tick_params(axis='y')
    # ax2.set_yticks([-scale, 0, scale]) # Adjust ticks if needed
    # ax2.set_yticklabels(['Sell', 'Hold', 'Buy']) # Simplify y-axis

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fancybox=True, ncol=2)

    # Final adjustments
    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.title(f'{metric_name} and Trading Decisions for {alg} in {task}', fontsize=14)
    plt.subplots_adjust(top=0.92) # Adjust top margin for title

    if save_dir is not None:
        save_path_img = osp.join(save_dir,f"Trading_Log_Viz_{task}.png")
        save_path_csv = osp.join(save_dir,f"trading_log_{task}.csv")
        try:
            plt.savefig(save_path_img)
            # Fill NaN with 0 before saving CSV for clarity
            trading_log.fillna(0).to_csv(save_path_csv)
            print(f"Saved visualization to {save_path_img}")
            print(f"Saved trading log to {save_path_csv}")
        except Exception as e:
            print(f"Error saving plot/log: {e}")
        plt.close() # Close the plot after saving
    else:
        plt.show()
