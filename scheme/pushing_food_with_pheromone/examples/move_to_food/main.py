import os


def main():
    from src import train, train2, replay, record

    save_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(save_dir, "2025_02_12_2059_f06ee2d0_log.pkl")

    # train(log_file_path)
    # replay(log_file_path)
    record(log_file_path)


if __name__ == '__main__':
    main()
