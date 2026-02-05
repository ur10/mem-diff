### Running the env
 - The env files to run for data collection are - `demo_pusht.py` `drawer_pusht.py` `multi_pusht.py`
 - To set up the required conda environment run - `conda env create -f conda_environment.yaml`
 - Example command for running the env - ` python3 drawer_pusht.py -o data/m1.zarr`
 -  - `drawer_pusht` -
  
      
    <img width="521" height="556" alt="image" src="https://github.com/user-attachments/assets/7a392cbc-eea7-4e99-81eb-8e8fe2598805" />


    - `multi_pusht` -
  
      
    <img width="521" height="556" alt="image" src="https://github.com/user-attachments/assets/3e6ec5b1-4141-4b31-a89d-f76b06acd9b4" />

 

  - `agent_counting` -
 
    <img width="521" height="556" alt="image" src="https://github.com/user-attachments/assets/c093b3a1-5e66-4265-88fc-d38895aa9d16"/>

### Training the diffusion policy with the data collected from the env
 - Set the env config file name - `export CONFIG_ENV_FILE_NAME=image_pusht_diffusion_policy_cnn_drawer.yaml` => This is for the 'multi_pusht' env.
 - Run the train script - `python train.py --config-dir=. --config-name=$CONFIG_ENV_FILE_NAME training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'`
