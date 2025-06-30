def genesis_example():
    from framework.backends.init_genesis import example_run
    example_run()


def viewer_example():
    from framework.backends import SimpleAnimatedBackend
    from framework.utils import GenericTkinterViewer
    from framework.config import Settings

    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    backend = SimpleAnimatedBackend(settings)
    viewer = GenericTkinterViewer(settings, backend)
    viewer.run()


def mujoco_example():
    from framework.backends import MujocoBackend
    from framework.utils import GenericTkinterViewer
    from framework.prelude import Settings, RobotLocation, Position

    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Robot.INITIAL_POSITION = [
        RobotLocation(0, 0, 0),
    ]
    settings.Food.INITIAL_POSITION = [
        Position(0, 2),
    ]

    backend = MujocoBackend(settings, render=True)
    viewer = GenericTkinterViewer(
        settings,
        backend,
    )
    viewer.run()


if __name__ == '__main__':
    mujoco_example()
