import time

from tensorboardX import SummaryWriter


class Logger:

    def __init__(self, spec):
        self.log_dir = spec["summary"]["destination"]
        self.add_on_episode_start(self.open)
        self.add_on_episode_end(self.close)
        self.log_subjects = spec["summary"]["models"]
        self.flush_interval = spec["summary"]["save_interval"]
        self.next_flush = time.time() + self.flush_interval

        self._histogram = []
        self._scalars = []

    def open(self):
        pass
    def log_scalar(self, log_name, val, step):
        if log_name in self.log_subjects:
            self._scalars.append(('data/' + log_name, val, step))
            # 'data/' + log_name, val, step

    def log_histogram(self, log_name, lst, step):
        if log_name in self.log_subjects:
            self._histogram.append((log_name, lst, step))
            #self.summary_writer.add_histogram(log_name, lst, step)

    def close(self):
        if time.time() > self.next_flush:
            self.summary_writer = SummaryWriter("runs/test")

            #for log_name, lst, step in self._histogram:
            #    self.summary_writer.add_histogram(log_name, lst, step)
            for log_name, val , step, in self._scalars:
                self.summary_writer.add_scalar(log_name, val, step)

            self._scalars = []
            self._histogram = []
            self.summary_writer.close()
            self.next_flush += self.flush_interval

