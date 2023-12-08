learning_rate = 0.01

start_epsilon = 0.9
end_epsilon = 0.05
epsilon_decay = 10000

gamma = 0.99

time_step = 0.02

batch_size = 16

think_interval = 2
life_time = int((60 / time_step) * 1)
area_size = 20

replay_buf_size = int(life_time / think_interval) * 5
