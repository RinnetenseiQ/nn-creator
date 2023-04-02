import tensorflow as tf


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        nonlocal trial_num
        nonlocal start_time
        if trial_num % 10 == 0:
            pass
        # print(f'Training model {trial_num + 1}/{max_trials}. Elapsed time: {datetime.datetime.now() - start_time}')
        trial_num += 1



