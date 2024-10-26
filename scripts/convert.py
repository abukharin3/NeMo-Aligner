

def submit_fake_nemo_job():
    new_ckpt_name="reinforce_70b_kl_0.01_LR_5e-7_64_trt_llm_True_use_reshard_True_pp_4_with_zarr_1_epochs_gbs128_saveint10_gdata_v3_step80.nemo"
    model_path="/lustre/fsw/portfolios/llmservice/users/abukharin/rlhf/results/70b_kl_0.01_LR_5e-7_64_trt_llm_True_use_reshard_True_pp_4_with_zarr_1_epochs_gbs128_saveint10_gdata_v3/actor_results/"
    base_model_path="/lustre/fs3/portfolios/llmservice/projects/llmservice_modelalignment_sft/rlhf/checkpoints/community/llama3.1/70b_instruct"
    #base_model_path="/lustre/fsw/portfolios/llmservice/users/geshen/share/8b_dpo-urban_3.002e-7-kl-1e-3-dpo-loss-rpo_fwd_kl-sft-weight-1e-5_megatron_gpt--val_loss=0.061-step=150-consumed_samples=38400-epoch=0/megatron_gpt--val_loss=0.061-step=150-consumed_samples=38400-epoch=0"
    ckpt="70b_kl_0.01_LR_5e-7_64_trt_llm_True_use_reshard_True_pp_4_with_zarr_1_epochs_gbs128_saveint10_gdata_v3_step80"
    cmd = f"""
    mkdir -p "{model_path}/checkpoints/{ckpt}/model_weights/";
    rm -r {model_path}/checkpoints/{ckpt}/optimizer*;
    mv {model_path}/checkpoints/{ckpt}/model.* {model_path}/checkpoints/{ckpt}/model_weights/;
    mv {model_path}/checkpoints/{ckpt}/common.pt {model_path}/checkpoints/{ckpt}/model_weights/;
    mv {model_path}/checkpoints/{ckpt}/metadata.json {model_path}/checkpoints/{ckpt}/model_weights/;
    cp {base_model_path}/*.yaml "{model_path}/checkpoints/{ckpt}/";
    cp {base_model_path}/1* "{model_path}/checkpoints/{ckpt}/";
    cp {base_model_path}/0* "{model_path}/checkpoints/{ckpt}/";
    cp {base_model_path}/3* "{model_path}/checkpoints/{ckpt}/";
    cp {base_model_path}/*.model "{model_path}/checkpoints/{ckpt}/";
    mv {model_path}/checkpoints/{ckpt} {model_path}/checkpoints/{new_ckpt_name};
    """
    print(cmd)

submit_fake_nemo_job()