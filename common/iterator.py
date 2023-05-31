from collections import deque
import torch


class PrefetchIterator:
    def __init__(self, kv, iterator, num_batches=10, intent=None, collate=None):
        self.kv = kv
        self.iterator = iterator
        self.num_batches = num_batches
        self.intent = intent
        self.collate = collate

    def __iter__(self):
        self.iter = iter(self.iterator)
        self.iter_done = False
        self.queue = deque()

        # fill queue with initial num_batches
        for _ in range(self.num_batches):
            self.loadnext()
            self.kv.advance_clock()

        return self

    def loadnext(self):
        try:
            item = next(self.iter)
            if self.collate:
                item = self.collate(item)
            self.queue.append(item)
            if self.intent:
                self.intent(item)
        except StopIteration:
            self.iter_done = True

    def __next__(self):
        if not self.iter_done:
            self.loadnext()

        if not self.queue:
            raise StopIteration

        return self.queue.popleft()

    def __len__(self):
        return len(self.iterator)
