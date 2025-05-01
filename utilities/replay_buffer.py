import numpy as np
import torch as th

class TransReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)

    def get_batch(self, batch_size):
        return self.get_truncated_episodes_batch(batch_size)

    def get_truncated_episodes_batch(self, batch_size):
        sample_range = len(self.buffer) - batch_size + 1
        start_indice = np.random.choice(sample_range, 1, replace=False)[0]
        batch_buffer = [self.buffer[i+start_indice] for i in range(batch_size)]
        return batch_buffer

    def add_experience(self, trans):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(trans)

    def clear(self):
        self.buffer = []

class TransReplayBufferPriority(object):
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.priorities = []

        self.alpha = 0.5
        self.beta = 0.5
        self.epsilon_for_priority = 1e-5

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        if len(self.buffer) > 0:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def add_experience(self, trans, init_priority=None):

        if len(self.buffer) >= self.size:
            self.offset()

        self.buffer.append(trans)
        if init_priority is None:
            init_priority = 1.0
        init_priority = max(init_priority, self.epsilon_for_priority)
        self.priorities.append(init_priority)

    def clear(self):
        self.buffer = []
        self.priorities = []

    def __len__(self):
        return len(self.buffer)

    def get_batch(self, batch_size):
        return self.get_truncated_episodes_batch(batch_size)

    def get_truncated_episodes_batch(self, batch_size):
        buffer_size = len(self.buffer)
        if buffer_size == 0:
            return [], [], []

        priorities_np = np.array(self.priorities, dtype=np.float32)
        scaled_priorities = priorities_np ** self.alpha
        total_p = np.sum(scaled_priorities)

        probs = scaled_priorities / total_p

        indices = np.random.choice(buffer_size, size=batch_size, p=probs, replace=False)

        N = buffer_size
        weights = (1.0 / (N * probs[indices])) ** self.beta
        weights /= weights.max()

        batch_buffer = [self.buffer[i] for i in indices]

        return batch_buffer, indices, weights

    def update_priorities(self, indices, td_errors):

        for idx, err in zip(indices, td_errors):
            if th.is_tensor(err):
                err = err.detach().cpu().item()
            new_p = abs(err) + self.epsilon_for_priority
            self.priorities[idx] = new_p
            print(f"Update priority of index {idx} to {new_p}")
        self.beta = min(1.0, self.beta + 0.0015)



class EpisodeReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)

    def get_batch(self, batch_size):
        length = len(self.buffer)
        indices = np.random.choice(length, batch_size, replace=False)
        batch_buffer = []
        for i in indices:
            batch_buffer.extend(self.buffer[i])
        return batch_buffer

    def add_experience(self, episode):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(episode)
