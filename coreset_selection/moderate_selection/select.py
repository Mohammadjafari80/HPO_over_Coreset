import numpy as np
from coreset_selection import CoresetSelection

class ModerateCoresetSelection(CoresetSelection):

    def get_median(self, features, targets):
        # get the median feature vector of each class
        num_classes = len(np.unique(targets, axis=0))
        prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)

        for i in range(num_classes):
            prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
            
        return prot


    def get_distance(self, features, labels):

        prots = self.get_median(features, labels)
        prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))

        num_classes = len(np.unique(labels))
        for i in range(num_classes):
            prots_for_each_example[(labels==i).nonzero()[0], :] = prots[i]
            
        distance = np.linalg.norm(features - prots_for_each_example, axis=1)

        return distance

    def get_prune_idx(self, rate, distance):

        low = 0.5 - rate / 2
        high = 0.5 + rate / 2

        sorted_idx = distance.argsort()
        low_idx = round(distance.shape[0] * low)
        high_idx = round(distance.shape[0] * high)

        ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))

        return ids

    def get_coreset(self):
        features, targets = self.get_features_targets()

        distance = self.get_distance(features, targets)

        rate = self.count / len(self.dataset)
        pruned_idx = self.get_prune_idx(rate, distance).tolist()
        coreset_idx = [c for c in range(len(self.dataset)) if c not in pruned_idx]

        if self.save:
            np.save(self.save_features_path, features)
            np.save(self.save_targets_path, targets)
            np.save(f'./coreset_indices/idx_{self.dataset_name}_{self.model_name}_{self.count}.npy', np.array(coreset_idx))

        return np.array(coreset_idx)