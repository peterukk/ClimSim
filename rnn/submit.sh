#!/bin/bash

source /data/piptorch/bin/activate

# export MPLBACKEND=TKAgg
export MPLBACKEND=Agg

#export WANDB_MODE=offline
#     use_water_loss=true w_wcon=3e7 \
# python train_rnn_rollout_torchscript_hydra.py --config-name=autoreg_LSTM_longwindows \
#     tr_data_fname=train_v4_rnn_nonorm_febtofeb_y1-7_stackedyear_subset2048.h5 \
#     tr_data_dir="/data/ClimSim/" \
#     val_data_dir="/data/ClimSim/" \
#     input_norm_per_level=true \
#     output_norm_per_level=true \
#     new_nolev_scaling=false \
#     use_energy_loss=true w_hcon=6.0e-06 \
#     use_water_loss=true w_wcon=2e7 \
#     use_precip_accum_loss=true w_precmse=1.2e14 \
#     use_bias_loss=false w_bias=4e1 \
#     model_type='LSTM' \
#     separate_radiation=true \
#     mp_mode=0 \
#     physical_precip=false \
#     loss_fn_type='huber' \
#     nneur=[144,144] \
#     lr_scheduler="OneCycleLR" \
#     scheduler_min_lr=3e-7 \
#     scheduler_max_lr=0.0015 \
#     scheduler_peak_epoch=4 \
#     scheduler_annealing='cos' \
#     scheduler_end_epoch=95 \
#     num_epochs=100 \
#     lr=0.0007 \
#     optimizer='soap' \
#     use_initial_mlp=true \
#     nh_mem=14 \
#     use_surface_memory=true \
#     rh_prune=false \
#     include_prev_outputs=true \
#     train_replay="mixed" \
#     val_replay="full" \
#     use_wandb=true \
#     rollout_schedule=[1,2,3,3,4,4,5,5,6,6,7,8,9,10,10,11] \
#     num_workers=4 \
#     val_epoch_start=0 
#     # gradual_mixing_end_epoch=40 \
    
    # cld_inp_transformation="sqrt"\
#  lr=0.0007

    # use_water_loss=false w_wcon=1.5e7 \


# python train_rnn_rollout_torchscript_hydra.py --config-name=autoreg_LSTM_longwindows \
#     tr_data_fname=train_v4_rnn_nonorm_febtofeb_y1-7_stackedyear_subset2048.h5 \
#     tr_data_dir="/data/ClimSim/" \
#     val_data_dir="/data/ClimSim/" \
#     input_norm_per_level=false \
#     output_norm_per_level=false \
#     new_nolev_scaling=false \
#     cld_inp_transformation="exp" \
#     use_energy_loss=true w_hcon=3.5e-05 \
#     use_water_loss=false w_wcon=1e7 \
#     use_precip_accum_loss=true w_precmse=1.5e14 \
#     use_bias_loss=false w_bias=1.34e2 \
#     use_cloudpath_loss=true w_cld=1.0e8 \
#     use_rh_loss=true w_rh=4.0 \
#     use_qn_positivity_loss=false w_qnpos=1e17 \
#     use_qv_positivity_loss=false w_qvpos=1e18 \
#     use_neg_precip_loss=true w_precip_neg=5e-5 \
#     model_type='GRU' \
#     mp_mode=-1 \
#     separate_radiation=false \
#     physical_precip=true \
#     include_q_input=true \
#     loss_fn_type='huber' \
#     nneur=[144,144] \
#     lr_scheduler="OneCycleLR" \
#     scheduler_min_lr=3e-7 \
#     scheduler_max_lr=0.0015 \
#     scheduler_peak_epoch=4 \
#     scheduler_annealing='cos' \
#     scheduler_end_epoch=95 \
#     num_epochs=30 \
#     lr=0.0007 \
#     optimizer='soap' \
#     use_initial_mlp=true \
#     nh_mem=16 \
#     use_surface_memory=false \
#     rh_prune=false \
#     include_prev_outputs=false \
#     use_wandb=true \
#     mp_autocast=false \
#     rollout_schedule=[2,2,3,3,4,4,5,5,6,6,6,6,6]  \
#     do_semi_online_training=false \
#     num_workers=4 \
#     val_epoch_start=0 #\
#     # model_file_checkpoint="GRU-Hidden_lr0.0007.neur144-144_xv4_mp-1_num33294.pt"
#     # train_replay="mixed" \
#     # val_replay="full" \

# existing_gasopt_file_lw="None" \
# existing_gasopt_file_lw="data/rrtmgp-data-lw-g128-210809_NN_GCM_NWP.nc" \
    # existing_gasopt_file_lw="data/rrtmgp-data-lw-g128-210809_NN_GCM_NWP.nc" \
    # existing_gasopt_file_sw="data/rrtmgp-data-sw-g112-210809_NN_GCM_NWP_absorption.nc" \
python train_rnn_rollout_torchscript_hydra.py --config-name=autoreg_LSTM_longwindows \
    tr_data_fname=train_v4_rnn_nonorm_febtofeb_y1-7_stackedyear_subset2048.h5 \
    tr_data_dir="/data/ClimSim/" \
    val_data_dir="/data/ClimSim/" \
    input_norm_per_level=false \
    output_norm_per_level=false \
    new_nolev_scaling=false \
    cld_inp_transformation="exp" \
    use_energy_loss=true w_hcon=2.5e-05 \
    use_water_loss=false w_wcon=1.0e7 \
    use_precip_accum_loss=true w_precmse=2.0e14 \
    use_bias_loss=false w_bias=1.34e2 \
    use_cloudpath_loss=true w_cld=1.0e8 \
    use_rh_loss=true w_rh=4.0 \
    use_qn_positivity_loss=false w_qnpos=1e17 \
    use_qv_positivity_loss=false w_qvpos=1e18 \
    use_neg_precip_loss=true w_precip_neg=5e-5 \
    model_type='physrad' \
    existing_gasopt_file_lw="data/rrtmgp-data-lw-g128-210809_NN_GCM_NWP.nc" \
    strat_temp_weight_factor=2.0 \
    scalar_weight_factor=6.0 \
    mp_mode=-1 \
    physical_precip=true \
    include_q_input=true \
    loss_fn_type='huber' \
    nneur=[128,128] \
    lr_scheduler="OneCycleLR" \
    scheduler_min_lr=3e-7 \
    scheduler_max_lr=0.0015 \
    scheduler_peak_epoch=4 \
    scheduler_annealing='cos' \
    scheduler_end_epoch=95 \
    num_epochs=100 \
    lr=0.0007 \
    optimizer='soap' \
    use_initial_mlp=true \
    nh_mem=16 \
    use_surface_memory=false \
    rh_prune=false \
    include_prev_outputs=true \
    use_wandb=true \
    mp_autocast=true \
    rollout_schedule=[2,2,3,3,4,4,4,4,4,4,4,4,4]  \
    do_semi_online_training=false \
    num_workers=4 \
    val_epoch_start=0 #\
    # model_file_checkpoint="physrad-Hidden_lr0.0007.neur128-128_xv4_mp-1_num42932.pt" \
    # save_loaded_model_and_quit=true
    # rollout_schedule=[2,2,3,3,4,4,5,5,6,6,6,6,6,6,6,6]  \
    # rollout_schedule=[2,2,3,3,4,4,4,4,4,4,4,4,4]  \

    # rollout_schedule=[1,2,3,3,4,4,5,5,6,6,7,8,9,10,10,11]  \
    # rollout_schedule=[1,2,3,3,4,4,5,5,5,5,5,5,5,5]  \
    # mp_autocast=false \
    # model_file_checkpoint="LSTM-Hidden_lr0.0007.neur144-144_xv4_mp0_num3348.pt" 
    # gradual_mixing_end_epoch=40 \

    # model_file_checkpoint="LSTM-Hidden_lr0.0007.neur144-144_xv4_mp0_num51981.pt" \
    # model_file_checkpoint="LSTM-Hidden_lr0.0007.neur144-144_xv4_mp0_num52322.pt" \
# rollout_schedule=[1,2,3,3,4,4,5,5,6,6,7,8,9,10,11,12] \




