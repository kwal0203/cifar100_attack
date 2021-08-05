import tensorflow_federated as tff
import attack as attacked_fedavg
import tensorflow_privacy
import tensorflow as tf
import numpy as np
import model
import data
import time
import csv
import sys


train, test, mal_train, mal_test = data.get_federated_datasets()
example_dataset = train.create_tf_dataset_for_client(client_id=train.client_ids[0])


"""
def get_keras_model():
  return tf.keras.Sequential([
    InputLayer(input_shape=[32, 32, 3], batch_size=1),
    Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.1),
    Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(units=1024, activation='relu'),
    Dropout(0.5),
    Dense(units=100),
    Softmax()])
"""

def create_tff_model():
    keras_model = model.create_resnet18()
    # keras_model = get_keras_model()
    input_spec = example_dataset.element_spec
    return tff.learning.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

client_update_function = attacked_fedavg.ClientProjectBoost(
      boost_factor=float(10),
      norm_bound=10,
      round_num=1)

query = tensorflow_privacy.GaussianSumQuery(1.0, 1.0)
query = tensorflow_privacy.NormalizedQuery(query, 10)

dp_agg_factory = tff.aggregators.DifferentiallyPrivateFactory(query)
iterative_process = attacked_fedavg.build_federated_averaging_process_attacked(
      model_fn=create_tff_model,
      model_update_aggregation_factory=None,
      client_update_tf=client_update_function,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
      server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0))

state = iterative_process.initialize()

def keras_evaluate(model, data, metric):
    metric.reset_state()
    for batch in data:
        preds = model(batch[0], training=False)
        metric.update_state(y_true=batch[1], y_pred=preds)
    return metric.result()

dummy_model = model.create_resnet18()
dummy_model.compile(
    optimizer=tf.keras.optimizers.SGD(0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

test_client = test.create_tf_dataset_for_client(train.client_ids[0])
test_mal_client = mal_test.create_tf_dataset_for_client(666)
metric = tf.keras.metrics.SparseCategoricalAccuracy()
# type_list = [tf.cast(0, tf.bool)] * 5
type_list = [tf.cast(0, tf.bool)] * 4
type_list += [tf.cast(1, tf.bool)]
mal_list = [mal_train.create_tf_dataset_for_client(666)] * 5
total = time.time()
running = 0.0
norms = [[], [], [], [], []]
round_stats = []
for i in range(5):
  round_time = time.time()
  _round_stats = []

  # Training
  data_ids = np.random.choice(a=train.client_ids[:5], size=5)
  data = [train.create_tf_dataset_for_client(i) for i in data_ids]
  state, metrics, output = iterative_process.next(state, data, mal_list, type_list)
  
  the_sum = 0.0
  for idx, val in enumerate(output):
    print(f"Norms: {val['weight_norm'].numpy():.3f}")
    norms[idx].append(val['weight_norm'].numpy())
    the_sum += val['weight_norm'].numpy()

  print(f"Average norm: {(the_sum / 5):.3f}")

  # Timing
  timer = time.time() - round_time
  average = running / (i+1)
  running += timer

  # Metrics
  loss = metrics['loss']
  train_acc = metrics['sparse_categorical_accuracy']
  state.model.assign_weights_to(dummy_model)
  test_acc = keras_evaluate(dummy_model, test_client, metric)
  mal_acc = keras_evaluate(dummy_model, test_mal_client, metric)
  print(f"----- Round: {i} -----")
  print(f"Loss:           {loss:.3f}")
  print(f"Train accuracy: {train_acc:.3f}")
  print(f"Test accuracy:  {test_acc:.3f}")
  print(f"Mal accuracy:   {mal_acc:.3f}")
  print(f"Average time:   {average:.2f}")
  _round_stats.append(i)
  _round_stats.append(loss)
  _round_stats.append(train_acc)
  _round_stats.append(test_acc.numpy())
  _round_stats.append(mal_acc.numpy())
  _round_stats.append(the_sum / 5)
  round_stats.append(_round_stats)

  with open('/home/kane/test/sun/cifar100/round_stats.csv', 'a') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(_round_stats)

  with open('/home/kane/test/sun/cifar100/round_norms.csv', 'a') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(round_norms)

for list in norms:
  print(f"Max: {max(list):.3f}")
  print(f"Min: {min(list):.3f}")
  print(f"Avg: {(sum(list) / len(list)):.3f}")
  print()
print(f"Training took {time.time()-total:.2f} seconds")

