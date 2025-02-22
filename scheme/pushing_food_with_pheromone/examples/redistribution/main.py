import os


def main():
    from src import train, train2, replay

    save_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(save_dir, "log.pkl")

    train(log_file_path)
    replay(log_file_path)


if __name__ == '__main__':
    main()
