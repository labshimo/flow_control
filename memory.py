import random 
import numpy as np


class PERProportionalMemory():
    def __init__(self, capacity, alpha=0.6, beta=0.4, e=0.01, total_steps=1000000, enable_is=False):
        self.capacity    = capacity
        self.tree        = SumTree(capacity)
        self.alpha       = alpha
        self.e           = e
        # importance sampling
        self.enable_is   = enable_is
        self.beta        = beta
        self.total_steps = total_steps
        # priority
        self.max_p       = 1

    def add(self, experience):
        self.tree.add(self.max_p, experience)

    def add_p(self, p, experience):
        self.tree.add(p, experience)

    def update(self, index, td_error):
        priority = (abs(td_error) + self.e) ** self.alpha
        self.tree.update(index, np.round(priority,4))

        if self.max_p < priority:
            self.max_p = priority

    def sample(self, batch_size, step):
        indexes  = []
        batchs   = []
        weights  = np.ones(batch_size, dtype='float32')

        if self.enable_is:
            beta = self.beta + (1 - self.beta) * step / self.total_steps
    
        total    = self.tree.total()
        segment  = total / (batch_size)
        
        i = 0
        while batch_size > len(batchs):
            # r    = section*i + random.random()*section + 0.0001
            a = segment * i
            b = segment * (i + 1)

            r = np.round(random.uniform(a, b), 5)
            idx, priority, experience = self.tree.get(r)
            
            if experience != 0:
                indexes.append(idx)
                batchs.append(experience)

                if self.enable_is:
                    weights[i] = (self.capacity * priority / total) ** (-beta)

                i += 1
        if self.enable_is:
            weights = weights / weights.max()

        return (indexes ,batchs, weights)
    
    def __len__(self):
        return self.tree.write

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1)
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

