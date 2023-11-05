import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, log_loss, accuracy_score
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import os

class ReferNeighbooringStrategy():
    def __init__(self, refer_features, refer_labels, num_class,
                    best_result = None,
                    alpha=None, beta=None, topk=None, 
                    main_metric='f1_weighted', 
                    batch_size=32, device='cpu'):
        '''
        :param main_metric: Main metric for referrence neighboring points
        :param alpha: List of coefficient of Cosine similarity probability 
        :param beta: List of coefficient of Euclidean distance probability
        :param topk: List of integer - number of neighbors
        '''
        self.refer_features = torch.FloatTensor(np.array(refer_features)).to(device)
        self.refer_labels = torch.FloatTensor(np.array(refer_labels)).to(device)
        self.main_metric = main_metric
        self.alpha = alpha 
        self.beta = beta
        self.topk = topk
        self.device = device
        self.num_class = num_class
        if best_result is not None:
            self.best_result = best_result
        else:
            self.best_result = (-1, -1, -1, -1) # (alpha, beta, topk, metric_score)
        self.batch_size = batch_size

    def flatten_ll(self, ll: list):
        return [item for l in ll for item in l]

    def fit(self, model, features, gold_labels, result_path, flatten=True, softmax=False):
        model.to(self.device)
        model.eval()
        
        result_file = open(result_path, 'w')
        if flatten:
            features = np.array(self.flatten_ll(features), dtype=np.float)
            gold_labels = np.array(self.flatten_ll(gold_labels), dtype=np.float)
        else:
            features = np.array(features, dtype=np.float)
            gold_labels = np.array(gold_labels, dtype=np.float)

        # Remove zero tensor
        nonzero_idx = np.sum(features, axis=-1) != 0
        features = features[nonzero_idx]
        gold_labels = gold_labels[nonzero_idx]

        dataset = TensorDataset(torch.FloatTensor(features), torch.LongTensor(gold_labels))
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        params_set = []
        for a in self.alpha:
            for b in self.beta:
                for k in self.topk:
                    params_set.append((a, b, k))

        for a, b, k in tqdm(params_set):
            with torch.no_grad():
                y_logits, y_preds = [], []
                for _, data in enumerate(dataloader): 
                    feature, y = data 
                    feature = feature.to(self.device)

                    # Infer model
                    y_hat_model = model(feature, return_proba=True)
                    y_proba_model = F.softmax(y_hat_model, dim=-1) if softmax else y_hat_model

                    # Refer to set of vectors by cosine similarity
                    y_proba_cos = self.get_proba_refer_by_cosine_similarity(
                                        feature, 
                                        self.refer_features, 
                                        self.refer_labels, 
                                        self.num_class, 
                                        k)

                    # Refer to set of vectores by euclidean distance
                    y_proba_ecd = self.get_proba_refer_by_euclidean_distance(
                                        feature, 
                                        self.refer_features, 
                                        self.refer_labels, 
                                        self.num_class, 
                                        k)

                    # Compute combination probability
                    y_proba = a * y_proba_cos + b * y_proba_ecd + (1 - a - b) * y_proba_model
                    y_hat = y_proba.argmax(dim=-1)

                    y_preds.append(y_hat.cpu().numpy())
                    y_logits.append(y.cpu().numpy())
                y_logits, y_preds = np.concatenate(y_logits, axis=0), np.concatenate(y_preds, axis=0)
                if self.main_metric == 'accuracy':
                    score = accuracy_score(y_logits, y_preds)
                else:
                    score = f1_score(y_logits, y_preds, average=self.main_metric[3:])
                if score > self.best_result[-1]:
                    self.best_result = (a, b, k, score)
                
            result_file.write(f'| Alpha = {a:.2f} | Beta = {b:.2f} | Top k = {k} | {self.main_metric} = {score:.4f} |\n')
                        
        result_file.write(f'=================================================================\n')
        result_file.close()

    def predict(self, model, batch, softmax=False):
        alpha, beta, k, _ = self.best_result
        batch = batch.squeeze().to(self.device)
        if batch.dim() == 1: batch = batch.unsqueeze(0)

        # Infer model
        y_hat_model = model(batch, return_proba=True)
        y_proba_model = F.softmax(y_hat_model, dim=-1) if softmax else y_hat_model

        if alpha == -1 or beta == -1 or k == -1:
            return y_proba_model.argmax(dim=-1)

        # Refer to set of vectors by cosine similarity
        y_proba_cos = self.get_proba_refer_by_cosine_similarity(
                            batch, 
                            self.refer_features, 
                            self.refer_labels, 
                            self.num_class, 
                            k)

        # Refer to set of vectores by euclidean distance
        y_proba_ecd = self.get_proba_refer_by_euclidean_distance(
                            batch,
                            self.refer_features,
                            self.refer_labels, 
                            self.num_class, 
                            k)

        # Compute combination probability
        y_proba = alpha * y_proba_cos + beta * y_proba_ecd + (1 - alpha - beta) * y_proba_model
        y_preds = y_proba.argmax(dim=-1)
        return y_preds
    
    def get_proba_refer_by_cosine_similarity(self, src: torch.tensor, trg: torch.tensor, trg_labels: torch.tensor, 
                                        num_class: int=2, topk: int=1):
        '''
        :param src: single vector [M, feature_dim]
        :param trg: set of vector [N, feature_dim]
        :return: probability of label [M, num_class]
        '''
        result = []
        for i in range(src.shape[0]):
            cos_values = F.cosine_similarity(src[i:i+1], trg)
            _, indices = cos_values.topk(k=topk, dim=-1, largest=True)
            y_hat = trg_labels[indices]
            votes = torch.tensor([y_hat[y_hat == i].shape[0] for i in range(num_class)], dtype=torch.float).to(self.device)
            proba = F.softmax(votes, dim=-1)
            result.append(proba)
        return torch.concat(result, dim=0).reshape(src.shape[0], num_class)

    def get_proba_refer_by_euclidean_distance(self, src: torch.tensor, trg: torch.tensor, trg_labels: torch.tensor, 
                                        num_class: int=2, topk: int=1):
        cdist = torch.cdist(src, trg, p=2)
        _, indices = cdist.topk(topk, dim=-1, largest=False)
        result = []
        for i in range(indices.shape[0]):
            y_hat = trg_labels[indices[i]]
            votes = torch.tensor([y_hat[y_hat == i].shape[0] for i in range(num_class)], dtype=torch.float).to(self.device)
            proba = F.softmax(votes, dim=-1)
            result.append(proba)
        return torch.concat(result, dim=0).reshape(src.shape[0], num_class)

    def parameters(self):
        return [self.refer_features.cpu().numpy(), 
                self.refer_labels.cpu().numpy(), 
                self.num_class, 
                self.best_result]


