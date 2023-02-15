import numpy
from studyLib import optimizer, miscellaneous, wrap_mjc


def cmaes_optimize(
        generation: int,
        population: int,
        mu: int,
        env_creator: optimizer.EnvCreator,
        sigma: float = 0.3,
        centroid=None,
        max_thread: int = 4,
        minimalize: bool = True,
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.CMAES(env_creator.dim(), generation, population, mu, sigma, centroid, minimalize, max_thread)
    opt.optimize(env_creator)
    return opt.get_best_para(), opt.get_history()


def cmaes_optimize_server(
        generation: int,
        population: int,
        mu: int,
        sigma: float,
        centroid,
        env_creator: optimizer.EnvCreator,
        port: int = 52325,
        minimalize: bool = True,
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.ServerCMAES(port, env_creator.dim(), generation, population, mu, sigma, centroid, minimalize)
    opt.optimize(env_creator)
    return opt.get_best_para(), opt.get_history()


def _client_proc(
        proc_id: int,
        default_env_creator: optimizer.EnvCreator,
        address: str,
        port: int,
        buf_size: int,
        timeout: float = 60.0
):
    import time
    import datetime

    opt = optimizer.ClientCMAES(address, port, buf_size)
    print(f"Start THREAD{proc_id}")

    # window = miscellaneous.Window(640, 480)
    # camera = wrap_mjc.Camera((0, 350, 0), 1200, 0, 90)

    error_count = 0
    while True:
        t = datetime.datetime.now()
        print(f"[INFO({t})] THREAD{proc_id} is calculating")
        result, pe = opt.optimize(default_env_creator, timeout)
        # result, pe = opt.optimize_and_show(default_env_creator, window, camera, timeout)  # debug

        if result is optimizer.ClientCMAES.Result.Succeed:
            error_count = 0
            continue
        elif result is optimizer.ClientCMAES.Result.ErrorOccurred:
            if error_count >= 3:
                print(f"[ERROR({t})] THREAD{proc_id} failed to reconnect the server. : ", pe)
                break
            print(f"[WARNING({t})] THREAD{proc_id} is retrying.")
            time.sleep(1)
            error_count += 1
            continue
        elif result is optimizer.ClientCMAES.Result.FatalErrorOccurred:
            print(f"[ERROR({t})] THREAD{proc_id} face fatal error : ", pe)
            break


def cmaes_optimize_client(
        num_thread: int,
        default_env_creator: optimizer.EnvCreator,
        address: str,
        buf_size: int = 3072,
        port: int = 52325,
        timeout: float = 60.0
):
    from multiprocessing import Process

    process_list = []
    for proc_id in range(num_thread):
        process = Process(
            target=_client_proc,
            args=(proc_id, default_env_creator, address, port, buf_size, timeout)
        )
        process.start()
        process_list.append(process)

    for p in process_list:
        p.join()
