import os

import src


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(save_dir, "TMP_LOG.pkl")

    # src.train(log_file_path)
    # src.replay(log_file_path)
    # src.record(log_file_path)
    # src.analyze(log_file_path)
    src.test(log_file_path)


if __name__ == '__main__':
    main()
