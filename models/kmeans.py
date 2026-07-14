import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean


# def initialize(X, num_clusters):
#     """
#     initialize cluster centers
#     :param X: (torch.tensor) matrix
#     :param num_clusters: (int) number of clusters
#     :return: (np.array) initial state
#     """
#     # indices_row = np.random.choice(X.shape[1], num_clusters*X.shape[0], replace=False)
#     # np.random.seed(1)
#     # indices_row = np.sort(np.random.choice(X.shape[1], num_clusters, replace=False))
#
#     indices_row = np.linspace(0, X.shape[1] - 1, num=num_clusters, endpoint=True, retstep=False, dtype=int)
#     indices_row = np.tile(indices_row,8)
#     # print(indices_row)
#
#     # indices_row = np.array([100, 300, 500, 700]*8)
#     indices_col = np.arange(X.shape[0]).repeat(num_clusters)
#
#     initial_state = X[indices_col, indices_row,:].reshape(X.shape[0],num_clusters,-1)
#     return initial_state
#
#
# def kmeans(
#         X,
#         num_clusters,
#         distance='euclidean',
#         tol=1e-4,
# ):
#     """
#     perform kmeans
#     :param X: (torch.tensor) matrix
#     :param num_clusters: (int) number of clusters
#     :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
#     :param tol: (float) threshold [default: 0.0001]
#     :param device: (torch.device) device [default: cpu]
#     :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
#     """
#
#     if distance == 'euclidean':
#         pairwise_distance_function = pairwise_distance
#     elif distance == 'cosine':
#         pairwise_distance_function = pairwise_cosine
#     else:
#         raise NotImplementedError
#
#     # convert to float
#     X = X.float()
#
#     # initialize
#     initial_state = initialize(X, num_clusters)
#
#     while True:
#         dis = pairwise_distance_function(X, initial_state)
#         choice_cluster = torch.argmin(dis, dim=2)#------tidudiushi
#         initial_state_pre = initial_state.clone()
#         from torch_scatter import scatter_mean, scatter_add
#         initial_state = scatter_mean(X, choice_cluster, dim=1, dim_size=num_clusters)
#
#         # for index in range(num_clusters):
#         #     selected = torch.nonzero(choice_cluster == index).squeeze()
#
#         #     selected = torch.index_select(X, 0, selected)
#         #     initial_state[index] = selected.mean(dim=0)
#
#         center_shift = torch.sum(
#             torch.sqrt(
#                 torch.sum((initial_state - initial_state_pre) ** 2, dim=2)
#             ),1)
#         if torch.all(~(center_shift ** 2).gt(tol)):
#             break
#
#     return choice_cluster, initial_state

def initialize(X, num_clusters):
    """
    修复索引长度不匹配：为每个样本独立生成聚类中心索引
    X: 输入特征，形状 [B, N, D]（B=batch_size，N=token数，D=特征维度）
    num_clusters: 每个样本的聚类数
    返回：初始聚类中心，形状 [B, num_clusters, D]
    """
    B, N, D = X.shape  # 获取输入的批量、token数、特征维度
    device = X.device

    # 1. 为每个样本生成 num_clusters 个随机token索引（形状 [B, num_clusters]）
    indices_row = torch.randint(0, N, (B, num_clusters), device=device)
    # 2. 生成批量索引（形状 [B, num_clusters]），确保每个样本只选自身的token
    indices_col = torch.arange(B, device=device).unsqueeze(1).repeat(1, num_clusters)

    # 3. 索引并reshape，此时 indices_col 和 indices_row 形状完全一致（都是 [B*num_clusters]）
    initial_state = X[indices_col.flatten(), indices_row.flatten(), :].reshape(B, num_clusters, D)
    return initial_state


def kmeans(
    X,
    num_clusters,
    distance='cosine',
    tol=1e-3,
    max_iter=20,
):
    """
    GPU简化稳定版K-Means（无CPU传输，无复杂操作）
    Args:
        X: 输入特征 (batch_size, num_samples, feature_dim)
        num_clusters: 聚类中心数量
        distance: 距离度量方式（cosine优先）
        tol: 中心移动阈值
        max_iter: 最大迭代次数
    Returns:
        cluster_ids: 每个样本的聚类标签 (batch_size, num_samples)
        centers: 最终聚类中心 (batch_size, num_clusters, feature_dim)
    """
    batch_size, num_samples, feature_dim = X.shape
    num_clusters = max(1, min(num_clusters, num_samples))

    # 初始化中心（最简单的随机选择，避免索引错误）
    indices = torch.randint(0, num_samples, (batch_size, num_clusters), device=X.device)
    indices = torch.clamp(indices, 0, num_samples - 1)
    centers = torch.gather(X, dim=1, index=indices.unsqueeze(-1).repeat(1, 1, feature_dim))

    for _ in range(max_iter):
        # 距离计算（仅保留余弦距离，最稳定）
        if distance == 'cosine':
            X_norm = F.normalize(X, p=2, dim=-1)
            centers_norm = F.normalize(centers, p=2, dim=-1)
            distances = 1 - torch.bmm(X_norm, centers_norm.transpose(1, 2))
        else:
            X_expand = X.unsqueeze(2)
            centers_expand = centers.unsqueeze(1)
            distances = torch.norm(X_expand - centers_expand, p=2, dim=-1)

        # 分配标签
        cluster_ids = torch.argmin(distances, dim=-1)
        cluster_ids = torch.clamp(cluster_ids, 0, num_clusters - 1)

        # 更新中心
        new_centers = scatter_mean(
            X,
            cluster_ids.unsqueeze(-1).repeat(1, 1, feature_dim),
            dim=1,
            dim_size=num_clusters
        )

        # 填充空聚类
        mask = torch.isnan(new_centers).any(dim=-1)
        new_centers[mask] = centers[mask]

        # 收敛判断
        center_shift = torch.norm(new_centers - centers, p=2, dim=-1).mean(dim=-1)
        if torch.all(center_shift < tol):
            break

        centers = new_centers

    return cluster_ids, centers




def pairwise_distance(data1, data2):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=2)

    # 1*N*M
    B = data2.unsqueeze(dim=1)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2):
    # transfer to device

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

