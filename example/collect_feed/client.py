from environments import utils
from environments.collect_feed import EnvCreator

if __name__ == '__main__':
    def main():
        utils.cmaes_optimize_client(
            6,
            EnvCreator(),
            "localhost",
            3072,
            52325,
            180.0
        )


    main()
