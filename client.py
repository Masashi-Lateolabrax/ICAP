from environments.back_enemy import Environment
from studyLib import wrap_mjc, miscellaneous, optimizer


def proc(address, port, buf_size, proc_id):
    env = Environment(
        (0, 0), [], [], 0.0, 0,
        # wrap_mjc.Camera((0, 0, 0), 120, 0, 90),
        # miscellaneous.Window(1200, 800)
    )
    opt = optimizer.ClientCMAES(address, port, env, buf_size)
    res = None
    while res is None:
        print(f"Thread{proc_id} is calculating")
        res = opt.optimize()
    print(f"END {proc_id}: ", res)


if __name__ == '__main__':
    def main():
        from multiprocessing import Process

        process_list = []
        for i in range(4):
            process = Process(
                target=proc,
                args=("192.168.11.37", 52325, 1024 * 6, i)
            )
            process.start()
            process_list.append(process)

        for p in process_list:
            p.join()


    main()
