from environments import utils, back_enemy

if __name__ == '__main__':
    def main():
        init_env = back_enemy.Environment((0, 0), [[(0, 0)]], [(0, 0)], 0, 0)
        utils.cmaes_optimize_client(6, init_env, "localhost")


    main()
