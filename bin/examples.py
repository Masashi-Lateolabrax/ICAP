def genesis_example():
    from framework.backends.init_genesis import example_run
    example_run()


def mujoco_example():
    from framework.backends.init_mujoco import example_run
    example_run()


def print_environment():
    from framework.environment import create_environment
    from framework.config.settings import Settings

    settings = Settings()
    spec = create_environment(settings)
    print(spec.to_xml())


def viewer_example():
    from framework.backends import SimpleAnimatedBackend
    from framework.utils.generic_viewer import GenericTkinterViewer

    backend = SimpleAnimatedBackend(width=480, height=320)
    viewer = GenericTkinterViewer(backend, width=480, height=320)
    viewer.run()


if __name__ == '__main__':
    viewer_example()
