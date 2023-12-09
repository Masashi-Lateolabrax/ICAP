learning_rate = 0.01
gamma = 0.99
batch_size = 16

start_epsilon = 0.9
end_epsilon = 0.05
epsilon_decay = 10000

time_step = 0.02
num_of_epoc = 100

think_interval = 2
lifetime = int((60 / time_step) * 1)
area = 20

replay_buf_size = int(lifetime / think_interval) * 5
