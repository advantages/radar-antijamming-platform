# import tensorflow as tf
#
#
# scalars = []
# tensorboard_path = "./log/radar-game/dqn_3_3_det1/dqn_3_3_det1_events.out.tfevents.1695633346.luolab-rl.1163805.0"
# for vs in tf.compat.v1.train.summary_iterator(tensorboard_path):
#     print(vs)
#     for v in vs.summary.value:
#         if v.tag == 'Main/loss':
#             scalars.append(tf.make_ndarray(v.tensor))

from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

file_name = './log_new/radar-game/trpo_4_4_6_det1_1012/events.out.tfevents.1697074510.luolab-rl.1799554.0'

ea = event_accumulator.EventAccumulator(file_name)

ea.Reload()
print(ea.Tags())

print(ea.Scalars('test/reward'))

writer = SummaryWriter()

for event in ea.Scalars('test/reward'):
    if event.step > 10000000:
        break

    writer.add_scalar('test/reward', event.value, event.step)

writer.close()
