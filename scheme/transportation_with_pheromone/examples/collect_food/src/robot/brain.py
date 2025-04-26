import torch

from scheme.transportation_with_pheromone.lib.objects.robot import RobotInput, BrainInterface, BrainJudgement


class Timer:
    def __init__(self, interval):
        self.interval = interval
        self.time = 0

    def tick(self):
        self.time += 1
        if self.time >= self.interval:
            self.time = 0
            return True
        return False


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(6, 6),
            torch.nn.Tanh(),
            torch.nn.Linear(6, 6),
            torch.nn.GELU(),
            torch.nn.Linear(6, 5),
            torch.nn.Softmax(dim=0)
        )

    def forward(self, x: RobotInput):
        x = x.get()
        y = self.sequence(x)
        return y

    def set_para(self, para):
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data = torch.tensor(
                para[i:i + n], dtype=torch.float32
            ).reshape(p.data.shape)
            i += n

    def num_dim(self):
        res = 0
        for p in self.parameters():
            res += p.numel()
        return res


class Brain(BrainInterface):
    @staticmethod
    def get_dim() -> int:
        return NeuralNetwork().num_dim()

    def __init__(self, para, interval):
        self.timer = Timer(interval)
        self.state = BrainJudgement.STOP

        self.neural_network = NeuralNetwork()
        self.neural_network.set_para(para)

    def think(self, input_: RobotInput) -> BrainJudgement:
        if not self.timer.tick():
            return self.state

        res = self.neural_network.forward(input_)
        selected = int(torch.multinomial(res, 1))
        # selected = torch.argmax(res).item()
        self.state = BrainJudgement(selected)

        return self.state
