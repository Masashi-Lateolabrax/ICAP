def main():
    import pickle
    import io
    import environment
    from src.DistributedComputing import NetManager

    net_manager = NetManager("localhost", 53215, 10)

    builder = environment.EnvironmentBuilder()

    buf = io.BytesIO()
    pickle.dump(builder, buf)
    net_manager.send(0, buf)


if __name__ == '__main__':
    main()
