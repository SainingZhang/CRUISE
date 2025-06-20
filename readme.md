# CRUISE: Cooperative Reconstruction and Editing in V2X Scenarios using Gaussian Splatting

Haoran Xu, [Saining Zhang](https://sainingzhang.github.io/), Peishuo Li, Baijun Ye, Xiaoxue Chen, Huan-ang Gao, Jv Zheng, Xiaowei Song, Ziqiao Peng, Run Miao, Jinrang Jia, Yifeng Shi, Guangqi Yi, Hang Zhao, Hao Tang, Hongyang Li, Kaicheng Yu, [Hao Zhao](https://sites.google.com/view/fromandto) <br />

### Demo



https://github.com/user-attachments/assets/f89cc217-a542-459e-946f-816803e31482



### Environment

Configure Python environment
```
# conda environment
conda create -n V2Xsim python=3.8
conda activate V2Xsim

# CUDA 11.8
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirments.txt

# Install submodules
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/simple-waymo-open-dataset-reader

```

Configure environment for generating masks [GroudingDINO](https://github.com/IDEA-Research/GroundingDINO), and download the [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)



Configure Python environment of [TRELLIS](https://github.com/microsoft/TRELLIS)



### Dataset
- Download the original dataset: [DAIR-V2X-SPD](https://thudair.baai.ac.cn/coop-forecast)

  - Use `data_process.ipynb` for data pre-processing

  - Use `generate_mask.ipynb` to generate sky mask and ego mask
  
- Or you can directly download and use the processed synthetic dataset: (comming soon)

- Download the high-quality vehicle dataset for Relightable3DGaussian: (comming soon)


### Training
If you want to modify the training command, change the content in train.sh and specify the corresponding config.
```
./script/train.sh
```

### Rendering

Use following command to render.
```
python render.py --config configs/xxxx.yaml mode edit
```

### Generate Synthetic dataset

Use the following command to perform preliminary organization and packaging of the render data.
```
python generate_dataset.py
```

Then use the command below to merge the synthetic dataset with the original dataset for downstream tasks.
```
python append_dataset.py
```

### Downstream tasks
Please complete the corresponding downstream as shown in the corresponding document.

- Infrastructure ciew 3d object detection: BEVHeight: https://github.com/ADLab-AutoDrive/BEVHeight

- Vehicle ciew 3d object detection:Monolss: https://github.com/Traffic-X/MonoLSS

- Coolaborative view 3d object detection: https://github.com/AIR-THU/DAIR-V2X/tree/main/configs/vic3d-spd/late-fusion-image
