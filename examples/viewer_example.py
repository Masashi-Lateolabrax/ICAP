def viewer_example():
    from framework.backends import SimpleAnimatedBackend
    from framework.utils import GenericTkinterViewer
    from framework.prelude import Settings

    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    backend = SimpleAnimatedBackend(settings)
    viewer = GenericTkinterViewer(settings, backend)
    viewer.run()


if __name__ == '__main__':
    viewer_example()
