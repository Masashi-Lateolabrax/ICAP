from environments import utils
from environments.move_obstacle import EnvCreator
from studyLib import wrap_mjc, miscellaneous

if __name__ == "__main__":
    def main():
        env_creator = EnvCreator()
        env_creator.robot_pos = (0, 200)
        env_creator.obstacle_pos = (0, 0)
        env_creator.timestep = int(30 / 0.03333)

        width: int = 640
        height: int = 480
        scale: int = 2
        window = miscellaneous.Window(width * scale, height * scale)
        camera = wrap_mjc.Camera((0, 0, 0), 3000, 0, 90)

        for i in range(1, 21):
            weight = 100 * i
            print(
                weight,
                utils.show_mujoco_env(env_creator, weight, window, camera)
            )


    main()
