import tkinter as tk
import mujoco

from mujoco_xml_generator.utils import MuJoCoView

from scheme.pushing_food_with_pheromone.lib.world import WorldBuilder1


class App(tk.Tk):
    def __init__(self, width=640, height=480):
        super().__init__()

        self.title("World")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.view = MuJoCoView(self.frame, width, height)
        self.view.enable_input()
        self.view.pack()

        self.world, _ = WorldBuilder1(
            0.01, (width, height), 10, 10
        ).build()

        self.renderer = mujoco.Renderer(
            self.world.model, height, width, 3000
        )

        self.after(0, self.update)

    def update(self):
        self.world.calc_step()
        self.view.render(self.world.data, self.renderer, dummy_geoms=self.world.get_dummy_geoms())
        self.after(1, self.update)


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
