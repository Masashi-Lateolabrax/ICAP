from prelude import framework

from brain import Brain
from gui import App
from settings import Settings
from loss import Loss


def main():
    settings = Settings(Loss())

    brain = Brain()

    para = framework.Parameters(brain)
    task = framework.TaskGenerator(settings).generate(para, debug=False)

    app = App(settings, task, 3000)
    app.mainloop()


if __name__ == '__main__':
    main()
