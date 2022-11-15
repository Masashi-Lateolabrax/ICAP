from environments import utils, back_enemy_with_pheromone

if __name__ == '__main__':
    def main():
        utils.cmaes_optimize_client(
            6,
            back_enemy_with_pheromone.EnvCreator(),
            "localhost"
        )


    main()
