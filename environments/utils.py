import numpy
from studyLib import optimizer, miscellaneous, wrap_mjc


def show_mujoco_env(
        env: optimizer.MuJoCoEnvInterface,
        para,
        camera: wrap_mjc.Camera,
        width: int = 640,
        height: int = 480,
        scale: int = 2,
) -> float:
    window = miscellaneous.Window(width * scale, height * scale)
    window.set_recorder(miscellaneous.Recorder("result.mp4", 30, width, height))
    env.set_window(window)
    env.set_camera(camera)
    return env.calc(para)


def cmaes_optimize(
        generation: int,
        population: int,
        env: optimizer.EnvInterface,
        minimalize: bool = True
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.CMAES(env, generation, population, 0.3, minimalize)
    opt.optimize()
    return opt.get_best_para(), opt.get_history()


def cmaes_optimize_server(
        generation: int,
        population: int,
        env: optimizer.EnvInterface,
        port: int = 52325,
        minimalize: bool = True
) -> (numpy.ndarray, optimizer.Hist):
    opt = optimizer.ServerCMAES(env, generation, population, 0.3, minimalize)
    opt.optimize(port)
    return opt.get_best_para(), opt.get_history()


def cmaes_optimize_client(
        num_thread: int,
        init_env: optimizer.EnvInterface,
        address: str,
        buf_size: int = 3072,
        port: int = 52325,
):
    from multiprocessing import Process
    import copy

    process_list = []
    for proc_id in range(num_thread):

        def proc():
            env = copy.deepcopy(init_env)
            opt = optimizer.ClientCMAES(address, port, env, buf_size)
            print(f"Start THREAD{proc_id}")

            res = None
            while res is None:
                print(f"THREAD{proc_id} is calculating")
                res = opt.optimize()
            print(f"THREAD{proc_id} is close : ", res)

        process = Process(target=proc)
        process.start()
        process_list.append(process)

    for p in process_list:
        p.join()
