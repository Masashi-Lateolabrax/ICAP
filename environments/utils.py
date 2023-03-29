import numpy
import multiprocessing
from studyLib import optimizer, miscellaneous, wrap_mjc


def cmaes_optimize(
        generation: int,
        population: int,
        mu: int,
        env_creator: optimizer.EnvCreator,
        sigma: float = 0.3,
        centroid=None,
        cmatrix=None,
        max_thread: int = 4,
        minimalize: bool = True,
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.CMAES(
        env_creator.dim(), generation, population, mu, sigma, centroid, cmatrix, minimalize, max_thread
    )
    opt.optimize(env_creator)
    return opt.get_best_para(), opt.get_history()


def cmaes_optimize_server(
        generation: int,
        population: int,
        mu: int,
        env_creator: optimizer.EnvCreator,
        port: int = 52325,
        sigma: float = 0.3,
        centroid=None,
        cmatrix=None,
        minimalize: bool = True,
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.ServerCMAES(
        port, env_creator.dim(), generation, population, mu, sigma, centroid, cmatrix, minimalize
    )
    opt.optimize_all(env_creator)
    return opt.get_best_para(), opt.get_history()


def _client_proc(
        proc_id: int,
        default_env_creator: optimizer.EnvCreator,
        address: str,
        port: int,
        buf_size: int,
        timeout: float,
        retry: int,
        global_gen: multiprocessing.Value
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
        result, pe = opt.optimize(default_env_creator, global_gen, timeout)
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

        elif result is optimizer.ClientCMAES.Result.Timeout:
            print(f"[ERROR({t})] Connecting is timeout in THREAD{proc_id} : ", pe)
            if error_count >= retry:
                print(f"[ERROR({t})] THREAD{proc_id} failed to reconnect the server. : ", pe)
                break
            error_count += 1
            print(f"[WARNING({t})] Connecting is retrying in THREAD{proc_id}. ({error_count}/{retry})")
            time.sleep(1)
            continue

        elif result is optimizer.ClientCMAES.Result.ForceTerminated:
            print(f"[DEBUG({t})] THREAD{proc_id} has been terminated forcibly: ", pe)
            continue


def cmaes_optimize_client(
        num_thread: int,
        default_env_creator: optimizer.EnvCreator,
        address: str,
        buf_size: int = 3072,
        port: int = 52325,
        timeout: float = 10.0,
        retry: int = 6,
):
    global_gen = multiprocessing.Value("i", 0)
    process_list = []
    for proc_id in range(num_thread):
        process = multiprocessing.Process(
            target=_client_proc,
            args=(proc_id, default_env_creator, address, port, buf_size, timeout, retry, global_gen)
        )
        process.start()
        process_list.append(process)

    for p in process_list:
        p.join()
