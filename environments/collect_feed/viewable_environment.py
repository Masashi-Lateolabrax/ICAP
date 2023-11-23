import dataclasses
import mujoco

import studyLib.environment as s_env
import studyLib.wrap_mjc as s_mjc
import studyLib.miscellaneous as s_misc

from environments.collect_feed.environment import Environment, EnvCreator


@dataclasses.dataclass
class Parameter:
    camera: s_mjc.Camera
    score: float
    viewpoint: mujoco.MjrRect = mujoco.MjrRect(0, 0, 0, 0)


class ViewableEnvironment(s_env.ViewableEnvInterface):
    def __init__(self, env_creator: EnvCreator):
        self._environment = Environment(env_creator)
        self.show_pheromone_index = env_creator.pheromone.show_pheromone_index

        self.pheromone_panel = s_misc.PheromonePanels(
            self._environment.world.model,
            0, 0,
            env_creator.pheromone_cell_size,
            self._environment.world.pheromone_field[0].nx,
            self._environment.world.pheromone_field[0].ny,
            0.05
        )

    def calc_step(self) -> float:
        return self._environment.calc_step()

    def render_gl_framebuffer(self, parameter: Parameter = None):
        import colorsys

        model = self._environment.world.model

        ctx = model.get_ctx()
        scn = model.get_scene(parameter.camera)

        pheromone_field = self._environment.world.pheromone_field[self.show_pheromone_index]
        self.pheromone_panel.update(
            pheromone_field,
            lambda gas: colorsys.hsv_to_rgb(0.66 * (1.0 - gas), 1.0, 1.0)
        )

        model.draw_text(f"{parameter.score}", 0, 0, (1, 1, 1))

        mujoco.mjr_render(parameter.viewpoint, scn, ctx)
