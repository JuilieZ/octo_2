import tensorflow_datasets as tfds
import h5py
import numpy as np

class TrajectoryDataset(tfds.core.GeneratorBasedBuilder):
    """自定义的数据集构建器，用于加载trajectory.h5数据集。"""
    
    VERSION = tfds.core.Version("1.0.0")
    
    def _info(self):
        # 设置数据集的元数据，如特征类型
        return tfds.core.DatasetInfo(
            builder=self,
            description="A dataset of trajectories stored in an HDF5 file.",
            features=tfds.features.FeaturesDict({
                'state': tfds.features.Tensor(shape=(None, 46), dtype=tf.float32),
                'action': tfds.features.Tensor(shape=(None, 14), dtype=tf.float32),
                'reward': tfds.features.Tensor(shape=(None,), dtype=tf.float32),
                'next_state': tfds.features.Tensor(shape=(None, 46), dtype=tf.float32),
            }),
            supervised_keys=None,
        )
    
    def _split_generators(self, dl_manager):
        # 返回训练/测试等数据集的分割（此示例只有一个split）
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'h5file': 'path/to/trajectory.h5'},
            ),
        ]
    
    def _generate_examples(self, h5file):
        # 加载 HDF5 文件并生成数据
        with h5py.File(h5file, 'r') as f:
            # 假设 HDF5 文件中有 'state', 'action', 'reward', 'next_state' 等键
            states = f['states'][:]
            actions = f['actions'][:]
            rewards = f['rewards'][:]
            next_states = f['next_states'][:]

            for i in range(len(states)):
                yield i, {
                    'state': states[i],
                    'action': actions[i],
                    'reward': rewards[i],
                    'next_state': next_states[i],
                }

