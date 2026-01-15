# Reinforcement Learning-based in Joint space via Hybrid Action Presentation

### V1.0, Jan 15th, 2026
**Authors:** [Thanh Nguyen Canh](https://thanhnguyencanh.github.io/), Thanh Tuan Tran, [Xiem HoangVan](https://sites.google.com/site/xiemhoang/), [Nak Young Chong](https://www.jaist.ac.jp/robot/).

## 1. Project Structure

    HybrAJ/
    │
    ├── assets/                 Objects and robot configuration files
    ├── env/                    Reinforcement learning environments
    ├── env_eval/                    Reinforcement learning environments for evaluation
    ├── models/                 RL models (TD3, SAC, ...)
    ├── reward/                 Reward function definitions
    ├── related_works/            relicated related works
    ├── utils/                      Utility functions (normalizers, buffer, ...)
    └── train_td3_min_random_prob.py   best proposed models

# 2. Prerequisites
Install all the python dependencies:
```
pip install -r requirements.txt
```
### Training
The system supports two robot arms, **UR5e** and **UF850**, each with four discrete action modes (```0: reach, 1: pick, 2: move, 3: put```)

To train a reinforcement learning policy, run:
```bash
python3 train_td3_min_random_prob.py --action {0,1,2,3} --robot {ur5e,uf850}
```
Note: Select only one value for each argument from the options listed above.
Starting a new training will automatically reset all existing checkpoints, logs, and related training artifacts.

# 3. Building and examples


# 4. License

# 5. Citation

If you use this work in an academic work, please cite:
