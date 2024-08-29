import abc

from libs.optimizer import MjcTaskInterface


class BaseDataCollector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_episode_length(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def pre_record(self, task, time: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def record(self, task, time: int, evaluation: float):
        raise NotImplementedError()

    def run(self, task: MjcTaskInterface):
        from datetime import datetime

        episode = self.get_episode_length()
        time = datetime.now()
        for t in range(episode):
            if (datetime.now() - time).seconds > 1:
                time = datetime.now()
                print(f"{t}/{episode} ({t / episode * 100}%)")

            self.pre_record(task, t)
            evaluation = task.calc_step()
            self.record(task, t, evaluation)
