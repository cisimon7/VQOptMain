import sys
import math
import numpy as np
import torch as th
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset


def postprocess(path: str):
    dataset = np.load(path, allow_pickle=True)
    trajs_obs = dataset["observations"]
    trajs_act = dataset["actions"]
    trajs_x = dataset["x_best"]
    trajs_y = dataset["y_best"]

    trajs_dx = np.stack(list(map(
        lambda vals: np.diff(vals, axis=0),
        trajs_x
    )))
    trajs_dy = np.stack(list(map(
        lambda vals: np.diff(vals, axis=0),
        trajs_y
    )))

    trajs_ddx = np.stack(list(map(
        lambda vals: np.diff(vals, axis=0),
        trajs_dx
    )))
    trajs_ddy = np.stack(list(map(
        lambda vals: np.diff(vals, axis=0),
        trajs_dy
    )))

    trajs_len = []
    for (traj_ddx, traj_ddy, traj_dx, traj_dy) in zip(trajs_ddx, trajs_ddy, trajs_dx, trajs_dy):
        traj_curv, traj_len = [], []
        for (step_ddx, step_ddy, step_dx, step_dy) in  zip(traj_ddx, traj_ddy, traj_dx, traj_dy):
            step_len = np.sum(np.power(np.power(step_dx, 2) + np.power(step_dy, 2), 1/2))
            traj_len.append(step_len)
        trajs_len.append(traj_len)

    trajs_len = np.asarray(trajs_len)

    good_indices = list(map(
        lambda vals: np.argwhere(vals > 5e1).flatten().astype(int),
        trajs_len
    ))

    observations = np.vstack(list(map(
        lambda vals: vals[0][:-1][vals[1]],
        zip(trajs_obs, good_indices)
    )))
    actions = np.vstack(list(map(
        lambda vals: vals[0][vals[1]],
        zip(trajs_act, good_indices)
    )))
    x_bests = np.vstack(list(map(
        lambda vals: vals[0][vals[1]],
        zip(trajs_x, good_indices)
    )))
    y_bests = np.vstack(list(map(
        lambda vals: vals[0][vals[1]],
        zip(trajs_y, good_indices)
    )))

    return th.from_numpy(observations), th.from_numpy(actions), th.from_numpy(x_bests), th.from_numpy(y_bests)


class TrajDataset(Dataset):
    def __init__(self, obs, traj):
        self.inp, self.out = obs, traj

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inp = self.inp[idx]
        out = self.out[idx]
        return inp.float(), out.float()


from commons import seed_mch

def get_dataset(batch_size, test_ratio=0.2, num=None):
    seed_mch()

    data = sio.loadmat("./datasets/data/ying_yang_data.mat")
    obs1 = th.from_numpy(data["observations"]).float()
    traj1 = th.from_numpy(data["trajectories"]).float()

    obs2, _, x_bests, y_bests = postprocess("./datasets/data/highway_dataset_xybest_5000_500_2.npz")
    traj2 = th.hstack([x_bests, y_bests])
    
    obs = th.vstack([obs1, obs2])
    traj = th.vstack([traj1, traj2])
    
    len_obs = len(obs)
    len_traj = len(traj)

    train_obs, test_obs = th.split(
        obs, [math.ceil((1-test_ratio) * len_obs), math.floor(test_ratio * len_obs)]
    )
    train_traj, test_traj = th.split(
        traj, [math.ceil((1-test_ratio) * len_traj), math.floor(test_ratio * len_traj)]
    )
    
    if num is not None:
        train_obs, train_traj = train_obs[:num], train_traj[:num]
    
    train_dataset = TrajDataset(train_obs, train_traj)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, worker_init_fn=lambda id_: seed_mch())
    
    if test_ratio > 0:
        test_dataset = TrajDataset(test_obs, test_traj)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, worker_init_fn=lambda id_: seed_mch())
    else:
        test_loader = None

    mean, std = th.mean(obs), th.std(obs)
    mean_joined, std_joined = th.mean(th.hstack([obs.flatten(), traj.flatten()])), th.std(th.hstack([obs.flatten(), traj.flatten()]))
    np.savez("./datasets/data/stats", mean=mean.numpy(), std=std.numpy(), mean_joined=mean_joined.numpy(), std_joined=std_joined.numpy())

    return train_loader, test_loader, mean_joined, std_joined


def get_dataset_aug(aug_data_path, batch_size, test_ratio=0.2):
    data = sio.loadmat("./datasets/data/ying_yang_data.mat")
    obs1 = th.from_numpy(data["observations"]).float()
    traj1 = th.from_numpy(data["trajectories"]).float()

    obs2, _, x_bests, y_bests = postprocess("./datasets/data/highway_dataset_xybest_5000_500_2.npz")
    traj2 = th.hstack([x_bests, y_bests])

    obs = th.vstack([obs1, obs2])
    traj = th.vstack([traj1, traj2])

    obs_aug, traj_aug = [], []
    for data_path in aug_data:
        obs_, _, x_bests, y_bests = postprocess(data_path)
        traj_ = th.hstack([x_bests, y_bests])
        obs_aug.append(obs_)
        traj_aug.append(traj_)

    obs_aug = th.vstack(obs_aug)
    traj_aug = th.vstack(traj_aug)

    len_obs_aug = len(obs_aug)

    if not append:
        obs = th.empty((0,)+obs.size()[1:])
        traj = th.empty((0,)+traj.size()[1:])
    elif replace:
        obs = obs[len_obs_aug:]
        traj = traj[len_obs_aug:]

    obs = th.vstack([obs, obs_aug])
    traj = th.vstack([traj, traj_aug])

    len_obs = len(obs)
    len_traj = len(traj)

    train_obs, test_obs = th.split(obs, [math.ceil((1-test_ratio) * len_obs), math.floor(test_ratio * len_obs)])
    train_traj, test_traj = th.split(traj, [math.ceil((1-test_ratio) * len_traj), math.floor(test_ratio * len_traj)])

    train_dataset = TrajDataset(train_obs, train_traj)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


    if test_ratio > 0:
        test_dataset = TrajDataset(test_obs, test_traj)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    else:
        test_loader = None

    mean, std = th.mean(obs), th.std(obs)
    mean_joined, std_joined = th.mean(th.hstack([obs.flatten(), traj.flatten()])), th.std(th.hstack([obs.flatten(), traj.flatten()]))
    np.savez(f"./datasets/stats_new/stats_{len(aug_data)}+{aug_data_path[-1]}", mean=mean.numpy(), std=std.numpy(), mean_joined=mean_joined.numpy(), std_joined=std_joined.numpy())

    return train_loader, test_loader, mean_joined, std_joined
