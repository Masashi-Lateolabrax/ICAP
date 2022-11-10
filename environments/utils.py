import numpy
from studyLib import optimizer, miscellaneous, wrap_mjc


def show_mujoco_env(
        env: optimizer.MuJoCoEnvInterface,
        para,
        window: miscellaneous.Window,
        camera: wrap_mjc.Camera,
        width: int = 640,
        height: int = 480
) -> float:
    window.set_recorder(miscellaneous.Recorder("result.mp4", 30, width, height))
    return env.calc_and_show(para, window, camera)


def cmaes_optimize(
        generation: int,
        population: int,
        sigma: float,
        env: optimizer.MuJoCoEnvInterface,
        minimalize: bool = True,
        window_and_camera: (miscellaneous.Window, wrap_mjc.Camera) = None
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.CMAES(env.dim(), generation, population, sigma, minimalize)
    if window_and_camera is None:
        opt.optimize(env)
    else:
        window, camera = window_and_camera
        opt.optimize_with_recoding_min(env, window, camera)
    return opt.get_best_para(), opt.get_history()


def cmaes_optimize_server(
        generation: int,
        population: int,
        sigma: float,
        env: optimizer.MuJoCoEnvInterface,
        port: int = 52325,
        minimalize: bool = True,
        window_and_camera: (miscellaneous.Window, wrap_mjc.Camera) = None
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.ServerCMAES(port, env.dim(), generation, population, sigma, minimalize)
    if window_and_camera is None:
        opt.optimize(env)
    else:
        window, camera = window_and_camera
        opt.optimize_with_recoding_min(env, window, camera)
    return opt.get_best_para(), opt.get_history()


def _client_proc(proc_id: int, init_env: optimizer.EnvInterface, address: str, port: int, buf_size: int):
    import copy

    env = copy.deepcopy(init_env)
    opt = optimizer.ClientCMAES(address, port, buf_size)
    print(f"Start THREAD{proc_id}")

    res = None
    while res is None:
        print(f"THREAD{proc_id} is calculating")
        res = opt.optimize(env)
    print(f"THREAD{proc_id} is close : ", res)


def cmaes_optimize_client(
        num_thread: int,
        init_env: optimizer.EnvInterface,
        address: str,
        buf_size: int = 3072,
        port: int = 52325,
):
    from multiprocessing import Process

    process_list = []
    for proc_id in range(num_thread):
        process = Process(
            target=_client_proc,
            args=(proc_id, init_env, address, port, buf_size)
        )
        process.start()
        process_list.append(process)

    for p in process_list:
        p.join()
